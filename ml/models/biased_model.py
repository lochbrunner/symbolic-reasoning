import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class BiasedModeler(nn.Module):
    '''
    This model only has one biased vector layer.

    Use this as baseline in order to benchmark the more complex models.
    '''

    def __init__(self, rules_size):
        super(BiasedModeler, self).__init__()
        self.bias = Parameter(torch.zeros([rules_size], dtype=torch.float))
        self.bias.requires_grad_()

    def forward(self, ident, remainder=None, hidden=None):
        out = F.log_softmax(self.bias)
        return out, None, None

    def initial_hidden(self):
        return None

    def initial_remainder(self):
        return []
