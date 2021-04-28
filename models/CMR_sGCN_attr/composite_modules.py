import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
from itertools import chain
from utils.pytorch_misc import NormalizeScale
from allennlp.nn.util import masked_softmax, replace_masked_values
import time

class NodeAttendModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.lang_normalizer = NormalizeScale(kwargs['dim_word'], 5)
        self.mlp = torch.nn.Sequential(
                nn.Linear(kwargs['dim_v'], kwargs['dim_v']),
        )

    def forward(self, node_rep, feat, box_mask):
        node_rep = self.lang_normalizer(node_rep)
        query = self.mlp(node_rep).unsqueeze(1)
        att_out = feat * query
        att_out = masked_softmax(torch.sum(att_out, dim=-1), box_mask, dim=1)
        att_out += 1e-8
        return att_out


class TransferModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        lang_dim = kwargs['dim_word']
        self.lang_normalizer = NormalizeScale(lang_dim, 5)
        self.map_c = nn.Linear(lang_dim, kwargs['dim_edge'])

    def forward(self, relate_rep, feat, feat_edge, relation_mask, att_in):
        # Attend edge module
        batch_size = feat_edge.size(0)
        relate_rep = self.lang_normalizer(relate_rep)
        query = self.map_c(relate_rep)
        query = query.view(batch_size, 1, 1, -1).expand_as(feat_edge)
        relation_mask = relation_mask.to(query.device)
        elt_prod = query * feat_edge

        inner_result = torch.sum(elt_prod, dim=3) * relation_mask.float()
        weit_matrix = masked_softmax(inner_result, relation_mask, dim=-2)

        # Transfer module
        att_out_list = []
        for i in range(att_in.shape[1]):  # for each node
            att_out = torch.matmul(att_in[:,i,:].unsqueeze(1), weit_matrix).permute(0,2,1).squeeze(-1)
            att_out += 1e-8
            att_out_list.append(att_out)
        att_out_list = torch.stack(att_out_list).permute(1,0,2)

        return att_out_list


class DescribeModule(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, final_att, vision_feat_raw):
        batch_size = vision_feat_raw.size(0)
        mem_out = torch.bmm(final_att.unsqueeze(1), vision_feat_raw.permute(0,2,1))
        mem_out = mem_out.view(batch_size, -1)
        return mem_out  # visual feature


