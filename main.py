#!/usr/bin/env python3

import torch
from torch import nn


from generate import create_samples

samples, idents, classes = create_samples(level=2, noise=1)

print(f'samples: {samples}')
print(f'idents: {idents}')
print(f'classes: {classes}')


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.f = None
        self.o = None
        self.c = None

    def forward(self, x):
        pass
