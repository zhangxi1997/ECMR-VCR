from typing import Dict, List, Any
import torch
import torch.nn as nn
from torchvision.models import resnet
from torch.nn.modules import BatchNorm2d,BatchNorm1d
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
import numpy as np

from .CMR import CMR
from pygcn.layers import MyGCN
from utils.pytorch_misc import parse_check, parse_check_ans, gene_Adj
from torch.nn.parameter import Parameter
import math
from torch.nn.modules.module import Module

# image backbone code from https://github.com/rowanz/r2c/blob/master/utils/detector.py
def _load_resnet_imagenet(pretrained=True):
    # huge thx to https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/nets/resnet_v1.py
    backbone = resnet.resnet50(pretrained=pretrained)
    for i in range(2, 4):
        getattr(backbone, 'layer%d' % i)[0].conv1.stride = (2, 2)
        getattr(backbone, 'layer%d' % i)[0].conv2.stride = (1, 1)
    # use stride 1 for the last conv4 layer (same as tf-faster-rcnn)
    backbone.layer4[0].conv2.stride = (1, 1)
    backbone.layer4[0].downsample[0].stride = (1, 1)

    # # Make batchnorm more sensible
    # for submodule in backbone.modules():
    #     if isinstance(submodule, torch.nn.BatchNorm2d):
    #         submodule.momentum = 0.01

    return backbone

@Model.register("CMRsGCNAttribute")
class CMRsGCNAttribute(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 option_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 visualize: bool = False,  # visualize mode
                 residual_graph: bool = False,  # residual img in Image Graph
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(CMRsGCNAttribute, self).__init__(vocab)

        ###################################################################################################
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.visualize = visualize
        self.residual_graph = residual_graph

        model_kwargs = {
            'vocab': vocab,  # not sure unavailable
            'dim_v': 512,
            'dim_word': 512,  # 768, #bert embedding answer/question
            'dim_hidden': 1024,  # hidden state of seq2seq parser
            'dim_vision': 512,  # obj_reps.shape[2]
            'dim_edge': 256,  # edge embedding
            'cls_fc_dim': 512,  # 1024, #classifier fc dim
            'dropout_prob': 0.5,
            'device': self.device,
            'visualize': self.visualize,
            'residual': self.residual_graph,
        }
        model_kwargs_tosave = {k: v for k, v in model_kwargs.items() if k != 'vocab'}

        self.CMR = CMR(**model_kwargs)
        self.gcn_r2o = MyGCN(512, 512)
        self.gcn_answer_r2o = MyGCN(512, 512)

        ###################################################################################################
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.image_BN = BatchNorm1d(512)

        self.option_encoder = TimeDistributed(option_encoder)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=option_encoder.get_output_dim(),
            matrix_2_dim=option_encoder.get_output_dim(),
        )

        self.option_BN = torch.nn.Sequential(
            BatchNorm1d(1024)
        )
        self.query_BN = torch.nn.Sequential(
            BatchNorm1d(1024)
        )
        self.mem_a_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.mem_q_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )
        self.attended_q_BN = torch.nn.Sequential(
            BatchNorm1d(512)
        )

        final_dim = 1024*2 + 512 * 2 + 512
        if self.residual_graph:
            final_dim += 512

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Linear(final_dim, int(final_dim / 2)),
            torch.nn.ReLU(inplace=True),
        )
        self.final_BN = torch.nn.Sequential(
            BatchNorm1d(int(final_dim / 2))
        )
        self.final_mlp_linear = torch.nn.Sequential(
            torch.nn.Linear(int(final_dim / 2), 1)
        )

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)      

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]
 
        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
           row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        return span_rep, retrieved_feats

    def forward(self,
                det_features: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                question_node_idx: torch.LongTensor,
                question_relate_idx: torch.LongTensor,
                question_relate_os: torch.LongTensor,
                answers_node_idx: Dict[str, torch.LongTensor],
                answers_relate_idx: Dict[str, torch.LongTensor],
                answers_relate_os: Dict[str,torch.LongTensor],
                question_node_mask: torch.LongTensor,
                answers_node_mask: torch.LongTensor,
                question_relate_mask: torch.LongTensor,
                answers_relate_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param det_features: [batch_size, 2048]
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param question_node_idx: the node idx in question [bs, max_node_num, max_seq_len]
        :param question_relate_idx: the relate idx in question [bs, max_node_num, max_seq_len]
        :param question_relate_os: the node_id for the triplet in question [bs, max_relate_num, 2]
        :param answers_node_idx: the node idx in answer [bs, 4, max_node_num, max_seq_len]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        det_features = det_features[:,:max_len]
        batch_size = boxes.shape[0]

        ####################################################################3
        obj_reps = det_features
        obj_reps = self.obj_downsample(obj_reps)

        options, option_obj_reps = self.embed_span(answers, answer_tags, obj_reps)
        option_rep = self.option_encoder(options, answer_mask)  # (batch_size, 4, seq_len, emb_len(512))
        query, query_obj_reps = self.embed_span(question, question_tags, obj_reps)
        query_rep = self.option_encoder(query, question_mask)  # (batch_size, 4, seq_len, emb_len(512))

        # parse_check
        question_node_idx, question_relate_idx = parse_check(query_rep, question_node_idx, question_relate_idx)
        answers_node_idx, answers_relate_idx = parse_check_ans(option_rep, answers_node_idx, answers_relate_idx)

        #######################################################################
        ''' syntactic GCN '''
        # construct Adj Matrix
        adj_question = gene_Adj(batch_size, query_rep.shape[2], question_relate_idx, question_relate_os,\
                                question_node_idx, question_mask, ans = False)

        option_len = option_rep.shape[2]
        adj_answer = torch.zeros((4, batch_size, option_len, option_len), requires_grad=False)
        for a in range(4):
            adj_answer[a] = gene_Adj(batch_size, option_len, answers_relate_idx[:,a,:,:], answers_relate_os[:,a,:,:],\
                                     answers_node_idx[:,a,:,:], answer_mask[:,a,:], ans = True)

        # forward
        query_input = torch.mean(query_rep, dim=1)
        query_rep_gcn_o2r = self.gcn_r2o(query_input, adj_question.cuda())
        query_rep_gcn_add = query_rep_gcn_o2r + query_input
        query_rep_gcn = replace_masked_values(query_rep_gcn_o2r, question_mask[:,0,:][...,None],0)

        option_rep_gcn = torch.zeros(batch_size, 4, option_len, 512).cuda()
        option_rep_gcn_add = torch.zeros(batch_size, 4, option_len, 512).cuda()
        for a in range(4):
            option_input = option_rep.permute(1, 0, 2, 3)[a]
            option_rep_gcn_o2r = self.gcn_answer_r2o(option_input, adj_answer[a].cuda())
            option_rep_gcn[:, a, :, :] = option_rep_gcn_o2r
            option_rep_gcn_add[:, a, :, :] = option_rep_gcn_o2r + option_input
        option_rep_gcn = replace_masked_values(option_rep_gcn, answer_mask[...,None],0)


        #############################################
        # get the representation for node in question
        question_node_rep = torch.bmm(question_node_idx, query_rep_gcn_add)
        total_num = torch.sum(question_node_idx, dim=-1)
        for mini in range(query_rep.shape[0]):
            for i in range(question_node_idx.shape[1]):
                if total_num[mini][i] == 0:
                    total_num[mini][i] = -1
        question_node_rep = question_node_rep / total_num.unsqueeze(2)  # [bs, node_num, dim]

        # get the representation for node in answer
        ans_node_rep_list = []
        for i in range(4):
            ans_nod_rep = torch.bmm(answers_node_idx[:,i,:,:], option_rep_gcn_add[:,i,:,:])
            total_num = torch.sum(answers_node_idx[:,i,:,:], dim=-1)
            for mini in range(option_rep.shape[0]):
                for i in range(ans_nod_rep.shape[1]):
                    if total_num[mini][i] == 0:
                        total_num[mini][i] = -1
            ans_node_rep_list.append(ans_nod_rep / total_num.unsqueeze(2))
        answers_node_rep = torch.stack(ans_node_rep_list).cuda().permute(1,0,2,3)

        # get the representation for relate in question
        question_relate_os = question_relate_os.long()
        question_relate_rep = torch.bmm(question_relate_idx, query_rep_gcn_add)
        total_num = torch.sum(question_relate_idx, dim=-1)
        for mini in range(query_rep.shape[0]):
            for i in range(question_relate_idx.shape[1]):
                if total_num[mini][i] == 0:
                    total_num[mini][i] = -1
        question_relate_rep = question_relate_rep /total_num.unsqueeze(2)  # [bs, relate_num, dim]
        question_relate_rep = replace_masked_values(question_relate_rep, question_relate_mask[..., None],0)

        # get the representation for relate in answer
        answers_relate_os = answers_relate_os.long()
        answers_relate_rep_list = []
        for i in range(4):
            ans_relate_rep = torch.bmm(answers_relate_idx[:,i,:,:], option_rep_gcn_add[:,i,:,:]) # [bs, relate_num, dim]

            total_num = torch.sum(ans_relate_rep, dim=-1)
            for mini in range(option_rep.shape[0]):
                for r in range(ans_relate_rep.shape[1]):
                    if total_num[mini][r] == 0:
                        total_num[mini][r] = -1
            ans_relate_rep = ans_relate_rep / total_num.unsqueeze(2)
            ans_relate_rep = replace_masked_values(ans_relate_rep, answers_relate_mask[:, i, :][..., None], 0)

            answers_relate_rep_list.append(ans_relate_rep)
        answers_relate_rep = torch.stack(answers_relate_rep_list).cuda().permute(1,0,2,3)

        ########################################################################
        ''' cross-modal reasoning '''

        num_feat = boxes.shape[1] - 1  # delete the whole img
        relation_mask_no36 = np.zeros((boxes.shape[0], num_feat, num_feat))
        boxes = boxes[:, 1:, :].cpu().numpy()
        whole_img_feat = obj_reps.permute(0, 2, 1)[:, :, 0]
        XNMNet_vision = obj_reps.permute(0, 2, 1)[:, :, :-1]
        box_mask = box_mask[:, 1:]

        # construct edge matrix for the Image graph
        for mini in range(boxes.shape[0]):
            for i in range(num_feat):
                for j in range(i + 1, num_feat):
                    if boxes[mini][i, 0] < 0:
                        pass
                    else:
                        if boxes[mini][i, 0] > boxes[mini][j, 2] or boxes[mini][j, 0] > boxes[mini][i, 2] or \
                                        boxes[mini][i, 1] > boxes[mini][j, 3] or boxes[mini][j, 1] > boxes[mini][i, 3]:
                            pass
                        else:
                            relation_mask_no36[mini][i, j] = relation_mask_no36[mini][j, i] = 1
                relation_mask_no36[mini][i, i] = 1
        relation_mask = torch.from_numpy(relation_mask_no36).byte()

        # the question-to-vision CMR
        mem_q, others_q = self.CMR(question_node_rep, question_relate_rep, question_relate_os, question_relate_mask, \
                                          XNMNet_vision, relation_mask, box_mask, question_node_mask)
        mem_q = torch.stack([mem_q for i in range(4)]).permute(1, 0, 2)

        # the answer-to-vision CMR
        mem_a_list = []
        others_a_list = []
        for i in range(4):  # four answer
            mem_a, others_a = self.CMR(answers_node_rep[:,i,:,:], answers_relate_rep[:,i,:,:], answers_relate_os[:,i,:,:], \
                                              answers_relate_mask[:, i, :], XNMNet_vision, relation_mask, box_mask, answers_node_mask[:,i,:])
            mem_a_list.append(mem_a)
            others_a_list.append(others_a)
        mem_a = torch.stack(mem_a_list).permute(1, 0, 2)

        ########################################################################
        ''' the r2c attention '''
        qa_similarity = self.span_attention(
            query_rep.contiguous().view(query_rep.shape[0] * query_rep.shape[1], query_rep.shape[2], query_rep.shape[3]),
            option_rep.contiguous().view(option_rep.shape[0] * option_rep.shape[1], option_rep.shape[2], option_rep.shape[3]),
        ).view(option_rep.shape[0], option_rep.shape[1], query_rep.shape[2], option_rep.shape[2])
        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, query_rep))
        attended_q_pool = replace_masked_values(attended_q, answer_mask[..., None], -1e7).max(2)[0]  # out: [bs,4,dim]

        ########################################################################
        # option representation
        batch_size, num_options, padded_seq_len, _ = answers['bert'].shape
        option_rep = replace_masked_values(option_rep, answer_mask[...,None], 0)
        assert (options.shape == (batch_size, num_options, padded_seq_len, 1280))
        seq_real_length = torch.sum(answer_mask, dim=-1, dtype=torch.float)  # (batch_size, 4)
        seq_real_length = seq_real_length.view(-1,1)  # (batch_size * 4,1)

        # syntactic GCN residual part
        option_rep_new = torch.cat([option_rep, option_rep_gcn], dim=-1)  # concat
        option_rep_new = option_rep_new.sum(dim=2)
        option_rep_new = option_rep_new.view(batch_size * num_options, 1024)
        option_rep_new = option_rep_new.div(seq_real_length)
        option_rep_new = self.option_BN(option_rep_new)
        option_rep_new = option_rep_new.view(batch_size, num_options, 1024)

        # query representation
        batch_size, num_options, padded_seq_len, _ = question['bert'].shape
        query_rep_new = replace_masked_values(query_rep, question_mask[..., None], 0)
        assert (query.shape == (batch_size, num_options, padded_seq_len, 1280))
        seq_real_length = torch.sum(question_mask, dim=-1, dtype=torch.float)
        seq_real_length = seq_real_length.view(-1, 1)

        # syntactic GCN residual part
        query_rep_new_span = torch.cat([query_rep_new, query_rep_gcn.unsqueeze(1).expand(query_rep_new.shape)])
        query_rep_new_span = query_rep_new_span.sum(dim=2)
        query_rep_new_span = query_rep_new_span.view(batch_size * num_options, 1024)
        query_rep_new_span = query_rep_new_span.div(seq_real_length)
        query_rep_new_span = self.query_BN(query_rep_new_span)
        query_rep_new_span = query_rep_new_span.view(batch_size, num_options, 1024)

        query_option_cat = torch.cat((option_rep_new, query_rep_new_span),-1)
        assert (query_option_cat.shape == (batch_size,num_options, 512*4))

        #####################################################
        ' VCR classification '
        mem_q = self.mem_q_BN(mem_q.contiguous().view(batch_size * num_options, 512))
        mem_q = mem_q.view(batch_size, num_options, 512)
        mem_a = self.mem_a_BN(mem_a.contiguous().view(batch_size * num_options, 512))
        mem_a = mem_a.view(batch_size, num_options, 512)

        attended_q_pool = self.attended_q_BN(attended_q_pool.contiguous().view(batch_size * num_options, 512))
        attended_q_pool = attended_q_pool.view(batch_size, num_options, 512)
        concated_rep = torch.cat([query_option_cat, attended_q_pool, mem_a, mem_q], dim=-1)
        if self.residual_graph:
            concated_rep = torch.cat([concated_rep, whole_img_feat.unsqueeze(1).expand(mem_a.size())], dim=-1)

        concated_rep = self.final_mlp(concated_rep)
        concated_rep = concated_rep.view(batch_size * num_options, -1)
        concated_rep = self.final_BN(concated_rep)
        concated_rep = concated_rep.view(batch_size, num_options, -1)

        logits = self.final_mlp_linear(concated_rep)
        logits = logits.squeeze(2)

        class_probabilities = F.softmax(logits, dim=-1)
        if self.visualize:
            output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                           'others_q': others_q,
                           'others_a': others_a_list,
                           }
        else:
            output_dict = {"label_logits": logits, "label_probs": class_probabilities,}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]

        return output_dict

    def get_metrics(self,reset=False):
        return {'accuracy': self._accuracy.get_metric(reset)}




