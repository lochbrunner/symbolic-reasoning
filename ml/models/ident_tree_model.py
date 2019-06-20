
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentTreeModeler(nn.Module):
    '''
    Based on the subtree
    '''

    def __init__(self, ident_size, remainder_dim, hidden_dim, embedding_dim, rules_size):
        super(IdentTreeModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.out_dim = rules_size
        self.remainder_dim = remainder_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(input_size=embedding_dim +
                            remainder_dim, hidden_size=self.hidden_dim)
        self.linear_out = nn.Linear(embedding_dim, rules_size)

    def forward(self, ident, remainder=None, hidden=None):
        if remainder is None:
            remainder = torch.zeros([1, self.remainder_dim], dtype=torch.float)

        if hidden is None:
            hidden = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float),
                      torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))

        embeds = self.embeddings(ident).view((1, -1))

        inputs = torch.cat((embeds, remainder), 1)

        inputs = inputs.view(len(inputs), 1, -1)
        lstm_out, hidden = self.lstm(inputs, hidden)
        (out, remainder) = torch.split(
            lstm_out, split_size_or_sections=self.embedding_dim, dim=2)

        out = F.relu(self.linear_out(out))
        out = F.log_softmax(out, dim=2)

        return out, hidden, remainder.view(1, -1)

    def initial_hidden(self):
        return (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float),
                torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))

    def initial_remainder(self):
        return [torch.zeros([1, self.remainder_dim], dtype=torch.float)]
