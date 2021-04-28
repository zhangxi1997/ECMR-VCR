import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn import DataParallel
import numpy as np
from itertools import chain
from . import composite_modules as modules
from allennlp.nn.util import masked_softmax, replace_masked_values
from utils.pytorch_misc import NormalizeScale

class CMR(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph
             dim_word, # word embedding dim
             dim_hidden, # hidden of seq2seq
             dim_vision,
             dim_answer,
             dim_edge,
             cls_fc_dim,
             dropout_prob,
             device,
             visualize,
             residual,
        """
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.visualize = kwargs['visualize']
        self.device = kwargs['device']
        self.dim_edge = kwargs['dim_edge']

        self.fc_word = nn.Linear(self.dim_edge + self.dim_word, self.cls_fc_dim)
        self.fc_vision = nn.Linear(self.dim_vision, self.cls_fc_dim)

        self.map_vision_to_v = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.dim_vision, self.dim_v, bias=False),
        )

        self.map_two_v_to_edge = nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.dim_v * 2, self.dim_edge, bias=False),
        )

        # modules
        self.NodeAttendModule = getattr(modules, 'NodeAttendModule')(**kwargs)
        self.TransferModule = getattr(modules, 'TransferModule')(**kwargs)
        self.DescribeModule = getattr(modules, 'DescribeModule')(**kwargs)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.feat_normalizer = NormalizeScale(self.dim_v, 5)

    def forward(self, node_rep, relate_rep, relate_os, relate_mask, vision_feat, relation_mask, box_mask, node_mask):
        """
        Args:
            :param node_rep: [batch_size, node_num, emb_len(512)]
            :param relate_rep: [batch_size, relate_num, emb_len(512)]
            :param relate_os: [batch_size, relate_num, 2]
            :param vision_feat (batch_size, dim_vision, num_feat)
            :param relation_mask (batch_size, num_feat, num_feat)
        """
        batch_size = node_rep.shape[0]
        ###################################################################
        # feature processing vision_feat: [bs, dim, num_feat]
        vision_feat_raw = vision_feat
        assert vision_feat.shape[1] == self.dim_vision
        feat_inputs = vision_feat.permute(0, 2, 1)
        bs, n = feat_inputs.size(0), feat_inputs.size(1)
        feat_inputs = self.feat_normalizer(feat_inputs.contiguous().view(bs * n, -1)).view(bs, n, -1)
        feat_inputs = self.map_vision_to_v(feat_inputs)  # (batch_size, num_feat, dim_v)

        num_feat = feat_inputs.size(1)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3)
        feat_edge = self.map_two_v_to_edge(feat_edge)  # (batch_size, num_feat, num_feat, dim_edge)

        ###################################################
        # Node attention module
        find_out_atts = []
        for i in range(node_rep.shape[1]):  # for each node
            find_out = self.NodeAttendModule(node_rep[:, i, :], feat_inputs, box_mask)
            find_out_atts.append(find_out)
        find_out_atts = torch.stack(find_out_atts).permute(1, 0, 2)  # [bs, max_node_num, obj_num]
        find_out_atts = replace_masked_values(find_out_atts, node_mask[..., None], 0)
        att_t0 = find_out_atts.clone()

        # Edge attention module and Transfer module
        transfer_out_relates = []
        for i in range(relate_rep.shape[1]):
            transfer_out = self.TransferModule(relate_rep[:, i, :], feat_inputs, feat_edge, relation_mask, find_out_atts)
            transfer_out_relates.append(transfer_out)

        transfer_out_relates = torch.stack(transfer_out_relates).permute(1, 0, 2, 3)
        # transfer_out_relates: [bs, relate_num, node_num, num_obj]
        relate_mask = relate_mask.unsqueeze(2).expand(batch_size, relate_os.shape[1],node_rep.shape[1])
        relate_mask = relate_mask.unsqueeze(3).expand(batch_size, relate_os.shape[1], node_rep.shape[1],att_t0.shape[-1])
        transfer_out_relates = transfer_out_relates * relate_mask.float()
        att_t1_transfer = transfer_out_relates  # [bs, relate_num, obj_num]

        for mini in range(batch_size):
            for i in range(relate_os.shape[1]):
                subject = relate_os[mini, i, 1]
                object = relate_os[mini, i, 0]
                if subject != -1:
                    find_out_atts[mini,subject,:] = find_out_atts[mini,subject,:] + transfer_out_relates[mini,i,object,:]

        att_t1 = find_out_atts.clone()

        # Describe
        final_att = find_out_atts.max(1)[0]
        norm = torch.max(final_att, dim=1, keepdim=True)[0].detach()
        norm[norm <= 1] = 1
        final_att /= norm
        final_att = replace_masked_values(final_att, box_mask, 1e-7)
        mem = self.DescribeModule(final_att, vision_feat_raw)

        final_mem = self.fc_vision(mem)
        if self.visualize:
            others = {
                'final_att': final_att,}
        else:
            others = {
                'final_att': final_att[0],}

        return final_mem, others