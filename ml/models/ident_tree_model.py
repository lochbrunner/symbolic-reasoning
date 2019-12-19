
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class IdentTreeLstmModeler(nn.Module):
    '''
    Based on the subtree


    ident -> embedding -> fc  🡖            🡕   fc -> out
    remainder          ->     +     lstm    -> fc -> remainder
    hidden (hidden, cell)           🡕
    '''

    def __init__(self, ident_size, remainder_dim, hidden_dim, embedding_dim, rules_size):
        super(IdentTreeLstmModeler, self).__init__()
        # Trainable parameters
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.out_dim = rules_size
        self.remainder_dim = remainder_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        lstm_size = embedding_dim
        self.lstm = nn.LSTM(input_size=lstm_size, hidden_size=self.hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, rules_size)
        self.linear_remainder_out = nn.Linear(hidden_dim, remainder_dim, bias=False)
        self.linear_remainder_in = nn.Linear(remainder_dim, lstm_size)
        self.initial_lstm_cell = Parameter(torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))
        self.initial_lstm_hidden = Parameter(torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))
        self.initial_remainder_param = Parameter(torch.zeros([1, 1, self.remainder_dim], dtype=torch.float))

    def forward(self, ident, remainder=None, hidden=None):
        if remainder is None:
            remainder = self.initial_remainder()

        if hidden is None:
            hidden = self.initial_hidden()

        embeds = self.embeddings(ident).view((1, -1))

        inputs = embeds + F.relu(self.linear_remainder_in(remainder))
        inputs = inputs.view(len(inputs), 1, -1)

        lstm_out, hidden = self.lstm(inputs, hidden)
        remainder = F.relu(self.linear_remainder_out(lstm_out))

        out = F.relu(self.linear_out(lstm_out))
        out = F.log_softmax(out, dim=2)

        return out, hidden, remainder

    def initial_hidden(self):
        return (self.initial_lstm_cell, self.initial_lstm_hidden)

    def initial_remainder(self):
        return self.initial_remainder_param
