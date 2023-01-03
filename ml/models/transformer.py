import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import unittest

# from pycore import Symbol


class PolicyHead(nn.Module):
    def __init__(self, embedding_size, tagset_size):
        super(PolicyHead, self).__init__()

        self.linear = nn.Linear(embedding_size, tagset_size)

    def forward(self, x, p):
        x = self.linear(x)
        x *= p.unsqueeze(2).expand(x.shape)
        y = F.log_softmax(x, dim=2)
        return torch.transpose(y, 1, 2)


class ValueHead(nn.Module):
    def __init__(self, embedding_size: int, config):
        super(ValueHead, self).__init__()

    def forward(self, x):
        raise NotImplementedError()


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, max_depth: int):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        spread = 2
        self.max_pos = spread**max_depth

    def forward(self, x, o):
        # map zero to zero?

        # x: b,l,i
        # o: b,u,l
        b, l, i = x.shape
        _, u, _ = o.shape

        # Full cycle
        print(f'self.max_pos: {self.max_pos}')
        d = (2*np.pi) / float(self.max_pos)
        loops = math.ceil(i / u)

        o = np.transpose(o, (0, 2, 1))
        o = np.tile(o, (1, 1, loops))[:, :, :i]
        # print(o)
        # print(f'x: {x.shape}')

        # different wave-lengths for each two tiles (sin & cos)
        s = np.arange(d, 0.01+d*(math.ceil(loops/2)), d, dtype=float).repeat(u*2)[:i]
        o *= s
        o[:, :, 0::2] = np.sin(o[:, :, 0::2])
        o[:, :, 1::2] = np.cos(o[:, :, 1::2])
        # print(f's: {s.shape}')
        print(f'o: {o.shape}')
        # o *=
        # print(o)
        # print(f'loops: {loops}')
        # print(f'i: {i}')
        # print(f'u: {u}')
        # print(f'd, d*(math.ceil(loops/2)): {d} {d*(math.ceil(loops/2))}')
        # s = np.repeat(s, loops*2)[:i]

        o = torch.as_tensor(o)
        return o
        # return x + self.dropout(o)


class TreeTransformer(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, pad_token: int, hyper_parameter, **kwargs):
        '''tagset_size with padding'''

        super(TreeTransformer, self).__init__()

        self.config = {
            'embedding_size': 32,
            'dropout': 0.1,
            'encoder_layers': 1,
            'number_of_heads': 8,
            'activation': 'relu',
            'dim_feedforward': 2048
        }

        if isinstance(hyper_parameter, dict):
            self.config.update(hyper_parameter)
        else:
            self.config.update(vars(hyper_parameter))

        embedding_size = self.config['embedding_size']
        dropout = self.config['dropout']
        # num_props = Symbol.number_of_embedded_properties

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=embedding_size,
            # embedding_dim=embedding_size-(num_props if self.config['use_props'] else 0),
            padding_idx=pad_token
        )

        self.pos_encoder = PositionalEncoding(embedding_size, dropout)

        # Backbone
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=self.config['number_of_heads'],
            dim_feedforward=self.config['dim_feedforward'], dropout=dropout,
            activation=self.config['activation']), self.config['encoder_layers'])

        # Heads
        self.policy = PolicyHead(embedding_size=embedding_size, tagset_size=tagset_size)
        self.value = ValueHead(embedding_size=embedding_size, config=self.config)

    def forward(self, x, o, p):
        # x: idents
        # o: positional encoding
        # p: policy sign
        x = self.embedding(x) * math.sqrt(self.config['embedding_size'])
        x = self.pos_encoder(x, o)
        x = self.transformer_encoder()

        return self.policy(x, p), self.value(x)

    @property
    def device(self):
        return next(self.parameters()).device


class TestPositionalEncoding(unittest.TestCase):
    def test_waves(self):
        import matplotlib.pyplot as plt
        o = np.array([[
            [8, 4, 12, 2, 6, 5, 7],
            [4, 2, 6, 1, 3, 0, 0],
            [2, 1, 3, 0, 0, 0, 0],
        ]], dtype=float)

        # x = np.zeros((1, o.shape[2], 8), dtype=float)
        x = torch.zeros((1, o.shape[2], 8), dtype=float)

        encoding = PositionalEncoding(hidden_dim=16, dropout=0.0, max_depth=4)

        y = encoding(x, o)[0, :, :]
        print(f'y: {y.shape}')
        plt.plot(y)
        plt.show()

        # print(y)


if __name__ == '__main__':
    unittest.main()
