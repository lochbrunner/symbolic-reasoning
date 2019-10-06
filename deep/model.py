import torch
import torch.nn as nn
import torch.nn.functional as F

from .node import Node


def ident_to_id(node: Node):
    return ord(node.ident) - 97


class TreeTagger(nn.Module):
    '''N-ary tree LSTM based on paper
    "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
    '''

    def __init__(self, max_spread, vocab_size, tagset_size, embedding_size, hidden_size):
        super(TreeTagger, self).__init__()
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


class TrivialTreeTagger(nn.Module):
    ''' Children are feeding its parent
    '''

    def __init__(self, vocab_size, tagset_size, embedding_size, hidden_size):
        super(TrivialTreeTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        print(f'tagset_size: {tagset_size}')
        print(f'embedding_size: {embedding_size}')

        assert tagset_size == hidden_size, 'For now hidden and tagsize must be equal'

        # Inputs are the outputs of the children
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        # Combinges the accumulated lstm output of the child and the ident of it self
        self.combine = nn.Linear(hidden_size+embedding_size, hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, tagset_size)
        # Before the first lstm input come (needed to thread leaf nodes)
        self.lstm_init = nn.Parameter(torch.Tensor(1, tagset_size))
        self.tagset_size = tagset_size

    def forward(self, node: Node):
        ident = torch.tensor(ident_to_id(node), dtype=torch.long)
        embeds = self.word_embeddings(ident)

        childs_out = [self(child) for child in node.childs]
        # print(childs_out)
        # childs_out = torch.Tensor([self(child) for child in node.childs])
        if len(node.childs) > 0:
            childs_out = torch.stack([self(child) for child in node.childs])
            # print(childs_out.size())
            # print(self.lstm_init.size())
            lstm_seq = torch.cat((self.lstm_init, childs_out), 0)
        else:
            lstm_seq = self.lstm_init
        lstm_out, _ = self.lstm(lstm_seq.view(-1, 1, self.tagset_size))
        lstm_out = torch.index_select(
            lstm_out, 0, torch.tensor([lstm_out.size()[0]-1])).view(-1)

        # print(f'lstm_out: {lstm_out.size()}')
        # print(f'embeds: {embeds.size()}')
        # print(embeds.size())
        # print(torch.cat((lstm_out.view(-1), embeds), 0).size())
        combined = self.combine(torch.cat((lstm_out, embeds), 0))
        tag_space = self.hidden2tag(combined)
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores[:]
