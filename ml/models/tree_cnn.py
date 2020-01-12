import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.symbol_builder import SymbolBuilder
from dataset.transformers import Embedder


class TreeCnnLayer(nn.Module):
    def _create_index_tensor(self, spread, depth):
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

    def __init__(self, spread, depth, in_size, out_size):
        super(TreeCnnLayer, self).__init__()

        self.index_tensor = self._create_index_tensor(spread, depth)
        mask = torch.Tensor(spread+2, in_size, out_size)
        stdv = 0.5
        nn.init.uniform_(mask, -stdv, stdv)
        self.mask = nn.Parameter(mask)
        self.bias = nn.Parameter(torch.zeros(out_size))

    def forward(self, x):
        # m: mask (k,i,j)
        # s: index map (l -> k,l)
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
            self.bias[None, None, -1].expand(b, l, -1)
        return F.relu(y)


class TreeCnnSegmenter(nn.Module):

    def __init__(self, vocab_size, tagset_size, pad_token, spread, depth, hyper_parameter, **kwargs):
        super(TreeCnnSegmenter, self).__init__()
        # Config
        self.config = {
            'max_spread': 2,
            'embedding_size': 16,
            'spread': 2,
        }
        self.config.update(hyper_parameter)

        embedding_size = self.config['embedding_size']

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=embedding_size,
            padding_idx=pad_token
        )

        self.cnn = TreeCnnLayer(spread, depth, embedding_size, tagset_size)

    def forward(self, x, *args):
        x = self.embedding(x)
        y = self.cnn(x)
        # j must be second index: b,j,...
        return torch.transpose(y, 1, 2)

    def activation_names(self):
        return []
