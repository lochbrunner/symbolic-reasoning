import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.symbol_builder import SymbolBuilder
from dataset.transformers import Embedder

from iconv import IConv

from pycore import Symbol


def _create_index_tensor(spread, depth):
    '''This tensor returns the k indices for node of index l'''
    builder = SymbolBuilder.create(spread=spread, depth=depth)

    index_table = {}
    # Use the embedder unroll method as single source of truth for the indices
    for c, node in enumerate(Embedder.unroll(builder.symbol_ref)):
        index_table[id(node)] = c

    index_table[id(None)] = len(index_table)

    mask_index_table = np.zeros([len(index_table), spread+2])
    # Each entries mask are [parent, self, *childs]
    # If parent or childs do not exist use padding at the end
    for node in builder.traverse_bfs():
        self_index = index_table[id(node)]
        parent_index = index_table[id(node.parent)]
        if len(node.childs) != 0:
            child_indices = [index_table[id(c)] for c in node.childs]
        else:
            child_indices = [index_table[id(None)] for _ in range(spread)]
        mask_index_table[self_index] = [self_index, parent_index] + child_indices

    return torch.as_tensor(mask_index_table, dtype=torch.long)


class PyIConv(nn.Module):
    '''Deprecated'''

    def __init__(self, in_size, out_size, index_tensor):
        super(PyIConv, self).__init__()

        self.register_buffer('index_tensor', index_tensor)
        k = index_tensor.size(1)
        mask = torch.Tensor(k, in_size, out_size)
        stdv = 0.5
        nn.init.uniform_(mask, -stdv, stdv)
        self.mask = nn.Parameter(mask)
        self.bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, x, *args):  # pylint: disable=arguments-differ
        # m: mask (k,i,j)
        # s: index map (l,k -> l)
        # x: input (b,l,i)
        # => b(s) (b,k,l,i)
        # -- indices
        # b: batch index
        # l: node index
        # i: input index (e.g. embedding size for the first layer)
        # j: output index
        # k: kernel index  (parent, self, childs)
        # y_blj = Σ_ki x_bl{s(l)_k}i m_kij + b_j
        l = x.size(1)
        b = x.size(0)
        i = x.size(2)

        # create windowed input x_l -> x_kl
        # 1. expand x_l -> x_lĸ where ĸ has the same size as l but without changing the value
        # 2. gather x_lĸ
        x = x[:, None, :, :].expand(-1, l, -1, -1)
        s = self.index_tensor[None, :, :, None].expand(b, -1, -1, i)

        x = torch.gather(x, 2, s)
        # x: (b,l,k,i)

        # Mul add operation
        # y_blj = Σ_ki x_blki m_kij + b_j
        # 1. flatten k and i index -> n
        #    => y_blj = Σ_n x_bln m_nj + b_j
        # 2. perform matmul
        # 3. expand and add bias
        y = torch.matmul(torch.flatten(x, 2, 3), torch.flatten(self.mask, 0, 1)) + \
            self.bias[None, None, :].expand(b, l, -1)
        return y


class SequentialByPass(nn.Sequential):

    def forward(self, x, s):  # pylint: disable=arguments-differ
        for module in self._modules.values():
            if type(module) is IConv:
                x = module(x, s)
            else:
                x = module(x)
        return x


class TreeCnnUniqueIndices(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, kernel_size, hyper_parameter, **kwargs):
        super(TreeCnnUniqueIndices, self).__init__()
        # Config
        self.config = {
            'embedding_size': 32,
            'hidden_layers': 2,
            'dropout': 0.1,
            'use_props': True
        }
        self.config.update(hyper_parameter)

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
        self.cnn_end = IConv(embedding_size, tagset_size, kernel_size=kernel_size)

    def forward(self, x, s, *args):  # pylint: disable=arguments-differ
        # x: b,l,(e,props)
        e = x[:, :, 0].squeeze()
        e = self.embedding(e)
        if self.config['use_props']:
            props = x[:, :, 1:].type(torch.FloatTensor)
            x = self.combine(e, props)
        else:
            x = e
        x = self.cnn_hidden(x, s)
        x = self.cnn_end(x, s)
        # j must be second index: b,j,...
        y = F.log_softmax(x, dim=2)
        return torch.transpose(y, 1, 2)

    @staticmethod
    def activation_names():
        return []

    @property
    def device(self):
        return next(self.parameters()).device


class TreeCnnSegmenter(nn.Module):
    '''
    TODO:
     * use layer-normalisation
     * use bilinear instead of concat for feature combination (-> to inputs)
    '''

    def __init__(self, vocab_size, tagset_size, pad_token, spread, depth, hyper_parameter, **kwargs):
        super(TreeCnnSegmenter, self).__init__()
        # Config
        self.config = {
            'embedding_size': 32,
            'hidden_layers': 2
        }
        self.config.update(hyper_parameter)
        self.spread = spread
        self.depth = depth

        embedding_size = self.config['embedding_size']

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=embedding_size,
            padding_idx=pad_token
        )

        index_tensor = _create_index_tensor(spread, depth)

        def create_layer():
            return IConv(embedding_size, embedding_size, indices=index_tensor)

        self.cnn_hidden = nn.Sequential(*[layer for _ in range(self.config['hidden_layers'])
                                          for layer in [create_layer(), nn.LeakyReLU(inplace=True)]])
        self.cnn_end = IConv(embedding_size, tagset_size, indices=index_tensor)

    def forward(self, x, *args):
        x = self.embedding(x)
        x = self.cnn_hidden(x)
        x = self.cnn_end(x)
        # j must be second index: b,j,...
        y = F.log_softmax(x, dim=2)
        return torch.transpose(y, 1, 2)

    @staticmethod
    def activation_names():
        return []

    @property
    def device(self):
        return next(self.parameters()).device
