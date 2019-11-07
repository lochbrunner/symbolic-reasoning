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

    @staticmethod
    def hyper_parameter_names():
        return ['embedding_size']

    def __init__(self, vocab_size, tagset_size, hyper_parameter):
        super(TrivialTreeTagger, self).__init__()
        embedding_size = hyper_parameter['embedding_size']

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # For now hidden and tagsize must be equal
        hidden_size = tagset_size

        # Inputs are the outputs of the children
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        # Combines the accumulated lstm output of the child and the ident of it self
        self.combine = nn.Linear(hidden_size+embedding_size, hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, tagset_size)
        # Before the first lstm input come (needed to thread leaf nodes)
        lstm_init = torch.empty(1, tagset_size)
        nn.init.uniform_(lstm_init, -1, 1)
        self.lstm_init = nn.Parameter(lstm_init)
        self.tagset_size = tagset_size

    def forward(self, node: Node):
        ident = torch.tensor(ident_to_id(node), dtype=torch.long)
        embeds = self.word_embeddings(ident)

        if len(node.childs) > 0:
            childs_out = torch.stack([self(child) for child in node.childs])
            lstm_seq = torch.cat((self.lstm_init, childs_out), 0)
        else:
            lstm_seq = self.lstm_init
        lstm_out, _ = self.lstm(lstm_seq.view(-1, 1, self.tagset_size))
        lstm_out = torch.index_select(
            lstm_out, 0, torch.tensor([lstm_out.size()[0]-1])).view(-1)

        combined = self.combine(torch.cat((lstm_out, embeds), 0))
        tag_space = self.hidden2tag(combined)
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores[:]

    def introspect(self, node: Node):
        '''
        Infers the node and returns 'all' intermediate results
        '''
        ident = torch.tensor(ident_to_id(node), dtype=torch.long)
        embeds = self.word_embeddings(ident)

        if len(node.childs) > 0:
            childs_out = torch.stack([self(child) for child in node.childs])
            lstm_seq = torch.cat((self.lstm_init, childs_out), 0)
        else:
            lstm_seq = self.lstm_init

        lstm_out, _ = self.lstm(lstm_seq.view(-1, 1, self.tagset_size))
        lstm_out = torch.index_select(
            lstm_out, 0, torch.tensor([lstm_out.size()[0]-1])).view(-1)

        combined = self.combine(torch.cat((lstm_out, embeds), 0))
        tag_space = self.hidden2tag(combined)
        tag_scores = F.log_softmax(tag_space, dim=0)

        return {
            'lstm_seq': lstm_seq.view(-1).detach().numpy(),
            'lstm_out': lstm_out.detach().numpy(),
            'scores': tag_scores.detach().numpy()
        }

    def activation_names(self):
        return ['lstm_out', 'scores', 'lstm_seq']
