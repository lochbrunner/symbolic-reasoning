import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCellOwn(nn.Module):
    '''
    Own implementation of LSTM Cell
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCellOwn, self).__init__()
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

    def forward(self, x, hx, cx):
        # i = W_ii*x+b_ii + W_hi*h_(t-1) +
        i = torch.sigmoid(self.W_ii.t().matmul(x) +
                          self.b_ii.t() + self.W_hi.t().matmul(hx) + self.b_hi.t())

        f = torch.sigmoid(self.W_if.t().matmul(x) +
                          self.b_if.t() + self.W_hf.t().matmul(hx) + self.b_hf.t())

        o = torch.sigmoid(self.W_io.t().matmul(x) +
                          self.b_io.t() + self.W_ho.t().matmul(hx) + self.b_ho.t())
        # Helping term
        g = torch.tanh(self.W_ig.t().matmul(x) +
                       self.b_ig.t() + self.W_hg.t().matmul(hx) + self.b_hg.t())

        # Cell
        cy = f*cx + i*g
        hy = o*torch.tanh(cy)

        return hy, cy


class LSTMCellRebuilt(nn.Module):
    '''
    From aten/src/ATen/native/RNN.cpp:390
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCellRebuilt, self).__init__()
        self.linear_ih = nn.Linear(input_size, hidden_size*4)
        self.linear_hh = nn.Linear(input_size, hidden_size*4)

    def forward(self, x, hx, cx):
        gates = self.linear_ih(x) + self.linear_hh(hx)
        chunked_gates = gates.chunk(4)
        ingate = torch.sigmoid(chunked_gates[0])
        forgetgate = torch.sigmoid(chunked_gates[1])
        cellgate = torch.tanh(chunked_gates[2])
        outgate = torch.sigmoid(chunked_gates[3])
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy


class Bilinear(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Bilinear, self).__init__()
        self.W_a = nn.Parameter(torch.Tensor(input_size, input_size))
        self.W_b = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))

    def forward(self, x, h):
        return self.W_a.t().matmul(x) + self.b.t() + self.W_b.t().matmul(h)


class LSTMCellOptimized(nn.Module):
    '''
    Own implementation of LSTM Cell
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCellOptimized, self).__init__()
        # Input gate
        self.i = Bilinear(input_size, hidden_size)

        # Forget gate
        self.f = Bilinear(input_size, hidden_size)

        # Output gate
        self.o = Bilinear(input_size, hidden_size)

        # G gate
        self.g = Bilinear(input_size, hidden_size)

    def forward(self, x, hx, cx):
        # i = W_ii*x+b_ii + W_hi*h_(t-1) +
        i = torch.sigmoid(self.i(x, hx))

        f = torch.sigmoid(self.f(x, hx))

        o = torch.sigmoid(self.o(x, hx))
        # Helping term
        g = torch.tanh(self.g(x, hx))

        # Cell
        cy = f*cx + i*g
        hy = o*torch.tanh(cy)

        return hy, cy


class LSTMCellOptimizedTwo(nn.Module):
    '''
    Own implementation of LSTM Cell
    '''

    def __init__(self, input_size, hidden_size):
        super(LSTMCellOptimizedTwo, self).__init__()
        # Input gate
        self.i = Bilinear(input_size, input_size)

        # Forget gate
        self.f = Bilinear(input_size, input_size)

        # Output gate
        self.o = Bilinear(input_size, input_size)
        self.s = Bilinear(input_size, input_size)
        # G gate
        self.g = Bilinear(input_size, input_size)

    def forward(self, x, c):
        # i = W_ii*x+b_ii + W_hi*h_(t-1) +
        i = torch.sigmoid(self.i(x, c))

        f = torch.sigmoid(self.f(x, c))

        o = torch.sigmoid(self.o(x, c))
        # Helping term
        g = torch.tanh(self.g(x, c))

        # Short-term
        # s = torch.sigmoid(self.s(x, s))

        # Cell
        c_next = f*c + i*g
        y = o*torch.tanh(c_next)

        return y, c_next
