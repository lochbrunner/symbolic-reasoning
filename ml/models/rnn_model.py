

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class RnnModeler(nn.Module):
    '''
    Simple RNN modeler using Linear Layers
    '''

    def __init__(self, ident_size, embedding_dim, horizontal_ramaining_dim, vertical_ramaining_dim, internal_dims, rules_size):
        super(RnnModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.linear_embedded_in = nn.Linear(embedding_dim, internal_dims[0])
        self.linear_h_remaining_in = nn.Linear(horizontal_ramaining_dim, internal_dims[0])
        self.linear_v_remaining_in = nn.Linear(vertical_ramaining_dim, internal_dims[0])

        self.linear_internals = [nn.Linear(internal_dims[i], internal_dims[i+1]) for i in range(len(internal_dims)-1)]

        self.linear_rules_out = nn.Linear(internal_dims[-1], rules_size)
        self.linear_h_remaining_out = nn.Linear(internal_dims[-1], horizontal_ramaining_dim)
        self.linear_v_remaining_out = nn.Linear(internal_dims[-1], vertical_ramaining_dim)

        self.initial_h_reamain_param = Parameter(torch.zeros([horizontal_ramaining_dim], dtype=torch.float))
        self.initial_remain_param = Parameter(torch.zeros([vertical_ramaining_dim], dtype=torch.float))

    def forward(self, ident, h_remain, v_remain):
        # Input
        embeds = self.embeddings(ident).view((1, -1))
        embeds_in = F.relu(self.linear_embedded_in(embeds))
        print(f'h_remain: {h_remain.size()}')
        print(f'v_remain: {v_remain.size()}')
        h_remaining_in = F.relu(self.linear_h_remaining_in(h_remain))
        v_remaining_in = F.relu(self.linear_v_remaining_in(v_remain))

        # Internal
        internal = embeds_in + h_remaining_in + v_remaining_in
        for linear_internal in self.linear_internals:
            internal = F.relu(linear_internal(internal))

        # Output
        out = F.relu(self.linear_rules_out(internal))
        h_remain_out = F.relu(self.linear_h_remaining_out(internal))
        v_remain_out = F.relu(self.linear_v_remaining_out(internal))

        print(f'h_remain_out: {h_remain_out.size()}')

        out, h_remain_out, v_remain_out

    def initial_hidden(self):
        return self.initial_h_reamain_param

    def initial_remainder(self):
        return self.initial_remain_param
