import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    '''
    Own implementation of LSTM Cell
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        # Input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b_ii = nn.Parameter(torch.Tensor(input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_hi = nn.Parameter(torch.Tensor(input_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b_if = nn.Parameter(torch.Tensor(input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_hf = nn.Parameter(torch.Tensor(input_size))

        # Output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b_io = nn.Parameter(torch.Tensor(input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_ho = nn.Parameter(torch.Tensor(input_size))

        # G gate
        self.W_ig = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b_ig = nn.Parameter(torch.Tensor(input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_hg = nn.Parameter(torch.Tensor(input_size))

    def forward(self, x, h, c):
        # i = W_ii*x+b_ii + W_hi*h_(t-1) +
        i = torch.sigmoid(self.W_ii.t().matmul(x) +
                          self.b_ii.t() + self.W_hi.t().matmul(h) + self.b_hi.t())

        f = torch.sigmoid(self.W_if.t().matmul(x) +
                          self.b_if.t() + self.W_hf.t().matmul(h) + self.b_hf.t())

        o = torch.sigmoid(self.W_io.t().matmul(x) +
                          self.b_io.t() + self.W_ho.t().matmul(h) + self.b_ho.t())
        # Helping term
        g = torch.tanh(self.W_ig.t().matmul(x) +
                       self.b_ig.t() + self.W_hg.t().matmul(h) + self.b_hg.t())

        # Cell
        c_next = f*c + i*g
        h = o*torch.tanh(c_next)

        return h, c_next


class LSTMCellOptimized(nn.Module):
    '''
    Own implementation of LSTM Cell
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCellOptimized, self).__init__()
        # Input gate
        self.W_ii = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_i = nn.Parameter(torch.Tensor(input_size))

        # Forget gate
        self.W_if = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_f = nn.Parameter(torch.Tensor(input_size))

        # Output gate
        self.W_io = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b_o = nn.Parameter(torch.Tensor(input_size))

        # G gate
        self.W_ig = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b_g = nn.Parameter(torch.Tensor(input_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, input_size))

    def forward(self, x, h, c):
        # i = W_ii*x+b_ii + W_hi*h_(t-1) +
        i = torch.sigmoid(self.W_ii.t().matmul(
            x) + self.b_i.t() + self.W_hi.t().matmul(h))

        f = torch.sigmoid(self.W_if.t().matmul(x) +
                          self.b_f.t() + self.W_hf.t().matmul(h))

        o = torch.sigmoid(self.W_io.t().matmul(x) +
                          self.b_o.t() + self.W_ho.t().matmul(h))
        # Helping term
        g = torch.tanh(self.W_ig.t().matmul(x) +
                       self.b_g.t() + self.W_hg.t().matmul(h))

        # Cell
        c_next = f*c + i*g
        h = o*torch.tanh(c_next)

        return h, c_next
