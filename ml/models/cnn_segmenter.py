import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from iconv import IConv

from pycore import Symbol


class SequentialByPass(nn.Sequential):

    def forward(self, x, s):  # pylint: disable=arguments-differ
        for module in self._modules.values():
            if type(module) is IConv:
                x = module(x, s)
            else:
                x = module(x)
        return x


class ValueHead(nn.Module):

    def __init__(self, embedding_size: int, config, kernel_size: int):
        super(ValueHead, self).__init__()
        self.hidden_size = config['value_head_hidden_size']

        self.cnn = SequentialByPass(
            IConv(in_size=embedding_size, out_size=self.hidden_size, kernel_size=kernel_size),
            nn.LeakyReLU(inplace=True),
            IConv(in_size=self.hidden_size, out_size=self.hidden_size, kernel_size=kernel_size),
            nn.LeakyReLU(inplace=True),
        )
        self.linear = nn.Linear(self.hidden_size, 2)  # Good or bad

    def forward(self, x, s):
        x = self.cnn(x, s)
        b, l, j = x.shape
        assert(self.hidden_size == j)
        # x: blj
        # blj -> (bl)j
        x = x.view([-1, j])
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = x.view([b, l, 2])
        # bl2 -> b2
        x = x.max(dim=1, keepdim=False).values
        return F.log_softmax(x, dim=1)


class PolicyHead(nn.Module):

    def __init__(self, embedding_size, kernel_size, tagset_size):
        super(PolicyHead, self).__init__()
        self.cnn = IConv(in_size=embedding_size, out_size=tagset_size, kernel_size=kernel_size)

    def forward(self, x, s, p):
        x = self.cnn(x, s)
        # negative policy indicates that the rule at that possition should not be applied
        # x_blj * p_bl = y_blj
        x *= p.unsqueeze(2).expand(x.shape)
        # j must be second index: b,j,...
        y = F.log_softmax(x, dim=2)
        return torch.transpose(y, 1, 2)


class TreeCnnSegmenter(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, kernel_size, hyper_parameter, **kwargs):
        '''tagset_size with padding'''

        super(TreeCnnSegmenter, self).__init__()
        # Config
        self.config = {
            'embedding_size': 32,
            'hidden_layers': 2,
            'dropout': 0.1,
            'use_props': True,
            'value_head_hidden_size': 8
        }
        if isinstance(hyper_parameter, dict):
            self.config.update(hyper_parameter)
        else:
            self.config.update(vars(hyper_parameter))

        embedding_size = self.config['embedding_size']

        num_props = Symbol.number_of_embedded_properties

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=embedding_size,
            # embedding_dim=embedding_size-(num_props if self.config['use_props'] else 0),
            padding_idx=pad_token
        )

        if self.config['use_props']:
            self.combine = nn.Bilinear(embedding_size, num_props, embedding_size)

        def create_layer():
            return IConv(embedding_size, embedding_size, kernel_size=kernel_size)

        self.cnn_hidden = SequentialByPass(*[layer for _ in range(self.config['hidden_layers'])
                                             for layer in [
                                                 create_layer(),
                                                 nn.LeakyReLU(inplace=True),
                                                 nn.Dropout(p=self.config['dropout'], inplace=False)]])

        # Heads
        self.policy = PolicyHead(embedding_size=embedding_size, kernel_size=kernel_size, tagset_size=tagset_size)
        self.value = ValueHead(embedding_size=embedding_size, kernel_size=kernel_size,
                               config=self.config)

    def forward(self, x, s, p, *args):  # pylint: disable=arguments-differ
        # p: b,l
        # x: b,l,(e,props)
        e = x[:, :, 0].squeeze()
        e = self.embedding(e)
        if self.config['use_props']:
            props = x[:, :, 1:].type(torch.FloatTensor).to(e.device)
            if e.ndim == 2:
                e = e.unsqueeze(0)
            x = self.combine(e, props)
        else:
            x = e
        x = self.cnn_hidden(x, s)

        return self.policy(x, s, p), self.value(x, s)

    @staticmethod
    def activation_names():
        return []

    @property
    def device(self):
        return next(self.parameters()).device
