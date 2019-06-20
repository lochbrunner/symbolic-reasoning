import torch
import torch.nn as nn
import torch.nn.functional as F


class RootIdentModeler(nn.Module):
    '''
    Based on the term's root ident

    Use this as baseline in order to benchmark the more complex models.
    '''

    def __init__(self, ident_size, embedding_dim, rules_size):
        super(RootIdentModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, rules_size)

    def forward(self, ident, remainder=None, hidden=None):
        embeds = self.embeddings(ident).view((1, -1))
        out = F.relu(self.linear(embeds))

        return out, None, None

    def initial_hidden(self):
        return None

    def initial_remainder(self):
        return []
