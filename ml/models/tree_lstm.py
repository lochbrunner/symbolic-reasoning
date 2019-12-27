import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from typing import List, Set, Dict, Tuple, Optional

from deep.node import Node


def ident_to_id(node: Node):
    return ord(node.ident) - 97


class TreeLstm(nn.Module):
    '''
    N-ary tree LSTM based on paper
    "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
    '''

    def __init__(self, max_spread, vocab_size, tagset_size, embedding_size, hidden_size):
        super(TreeLstm, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        self.gates_W_i = nn.Linear(embedding_size, 4*hidden_size)
        self.gates_W_h = nn.Linear(
            max_spread*hidden_size, 4*hidden_size, bias=False)

        self.hidden2tag = nn.Linear(embedding_size, tagset_size)

    def forward(self, node: Node):
        ident = torch.tensor(ident_to_id(node), dtype=torch.long)
        embeds = self.word_embeddings(ident)

        gates = self.gates_W_i()

        tag_space = self.hidden2tag(embeds)
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores[:]
