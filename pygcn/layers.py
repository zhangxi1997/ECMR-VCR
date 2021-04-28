import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

class MyGCN(Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
        """

    def __init__(self, in_features, out_features, bias=True, residual=False):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, batch_adj:torch.Tensor):

        with torch.no_grad():
            D = batch_adj.sum(-1, keepdim=True) + 1e-10
            D = torch.pow(D, -0.5)
            D[torch.isnan(D)] = 1.0

            batch_adj = batch_adj * D
            batch_adj = batch_adj * D.permute(0, 2, 1)

        h = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), self.weight.size(0), self.weight.size(1)))
        h = torch.bmm(batch_adj, h)
        if self.bias is not None:
            h += self.bias

        h = torch.nn.functional.relu(h)
        if self.residual:
            h += input

        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


x = torch.randn(10, 5, 200)
adj = torch.randn(10, 5, 5)
gcn = MyGCN(200, 4)

#print(gcn(x, adj))
