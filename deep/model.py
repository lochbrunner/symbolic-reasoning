import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from typing import List, Set, Dict, Tuple, Optional

from deep.node import Node


def ident_to_id(node: Node):
    return ord(node.ident) - 97


class TreeTagger(nn.Module):
    '''
    N-ary tree LSTM based on paper
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

    def __init__(self, vocab_size, tagset_size, device, hyper_parameter):
        super(TrivialTreeTagger, self).__init__()
        self.device = device
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

    def _cell(self, ident: torch.Tensor, lstm_seq: torch.Tensor):
        embeds = self.word_embeddings(ident)
        lstm_out, _ = self.lstm(
            lstm_seq.view(-1, 1, self.tagset_size).to(self.device))
        # print(f'lstm_out.size()[0]: {lstm_out.size()[0]}')
        lstm_out = torch.index_select(
            lstm_out, 0, torch.tensor([lstm_out.size()[0]-1])).view(-1)

        combined = self.combine(torch.cat((lstm_out, embeds), 0))
        tag_space = self.hidden2tag(combined)
        return F.log_softmax(tag_space, dim=0)

    def forward(self, node: Node):
        ident = torch.tensor(ident_to_id(node), dtype=torch.long)

        if len(node.childs) > 0:
            childs_out = torch.stack([self(child) for child in node.childs])
            lstm_seq = torch.cat((self.lstm_init, childs_out), 0)
        else:
            lstm_seq = self.lstm_init
        return self._cell(ident, lstm_seq)[:]

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


def extract_or(obj, field, default):
    if field in obj:
        return obj[field]
    else:
        return default


class TrivialTreeTaggerBatched(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, hyper_parameter):
        # print(f'tagset_size: {tagset_size}')
        super(TrivialTreeTaggerBatched, self).__init__()
        embedding_size = hyper_parameter['embedding_size']
        self.lstm_hidden_size = extract_or(hyper_parameter, 'lstm_hidden_size', 64)
        lstm_layers = extract_or(hyper_parameter, 'lstm_layers', 1)

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=embedding_size,
            padding_idx=pad_token
        )

        # LSTM
        # lstm_hidden_size = 100
        # lstm_layers = 1

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        lstm_h = torch.empty(lstm_layers, embedding_size,  # pylint: disable=no-member
                             self.lstm_hidden_size)
        nn.init.uniform_(lstm_h, -1., 1.)
        self.lstm_h = nn.Parameter(lstm_h)

        lstm_c = torch.empty(lstm_layers, embedding_size,  # pylint: disable=no-member
                             self.lstm_hidden_size)
        nn.init.uniform_(lstm_c, -1., 1.)
        self.lstm_c = nn.Parameter(lstm_c)

        # Combine ident with childs
        self.combine = nn.Linear(
            self.lstm_hidden_size+embedding_size, self.lstm_hidden_size)

        # To tag
        self.hidden_to_tag = nn.Linear(self.lstm_hidden_size, tagset_size)

    def _cell(self, r, c0, c1, s):
        c = torch.stack((c0, c1), dim=1)  # pylint: disable=no-member
        # batch x sequence x embedding
        c = self.embedding(c)
        # print(f'c: {c.size()}')
        # print(f'h: {self.lstm_h.size()}')
        c = rnn.pack_padded_sequence(c, s, batch_first=True, enforce_sorted=False)
        c, _ = self.lstm(c, (self.lstm_h, self.lstm_c))
        c, _ = rnn.pad_packed_sequence(c, batch_first=True)

        # Combine root with label
        c = F.relu(c)
        c = torch.split(c, 1, dim=1)[-1].view(-1, self.lstm_hidden_size)
        # print(f'c: {c.size()}')
        r = self.embedding(r)
        # print(f'r: {r.size()}')
        # x = torch.stack((r, c), dim=2)
        x = torch.cat((r, c), dim=1)
        x = self.combine(x)
        x = F.relu(x)

        x = self.hidden_to_tag(x)
        return x

    def forward(self, x, s):
        # For now only support spread 2 depth 1

        # x: batch x sequence
        # first child, second child, root
        # print(x.size())
        # print(f'l: {l}')
        c0, c1, r = [s.view(-1) for s in torch.split(x, 1, dim=1)]
        x = self._cell(r, c0, c1, s)
        x = F.log_softmax(x, dim=1)
        return x


class TrivialTreeTaggerUnrolled(TrivialTreeTagger):
    '''
    Assuming each node has the same structure.
    spread:2 depth:2
    '''

    # def __init__(self, vocab_size, tagset_size, hyper_parameter):
    #     super(TrivialTreeTaggerUnrolled, self).__init__(
    #         vocab_size, tagset_size, hyper_parameter)

    # def _batched_cell(self, ident: torch.Tensor, lstm_seq: torch.Tensor):
    #     batch_size = ident.size()[0]
    #     # print(f'batch_size: {batch_size}')
    #     embeds = self.word_embeddings(ident)
    #     lstm_out, _ = self.lstm(
    #         lstm_seq.view(batch_size, -1, self.tagset_size))
    #     # print(f'lstm_out: {lstm_out.size()}')

    #     # Get the last vector
    #     lstm_out = lstm_out[torch.arange(
    #         batch_size), [lstm_out.size()[0]-1]].view(batch_size, -1)
    #     # lstm_out = torch.index_select(
    #     #     lstm_out, 0, torch.tensor([lstm_out.size()[0]-1]).cuda()).view(-1)
    #     # print(f'lstm_out (selected): {lstm_out.size()}')
    #     # print(f'embeds: {embeds.size()}')
    #     concatened = torch.cat((lstm_out, embeds), 1)
    #     print(f'concatened: {concatened.size()}')
    #     combined = self.combine(concatened)
    #     print(f'combined: {combined.size()}')
    #     tag_space = self.hidden2tag(combined[:, -1])
    #     return F.log_softmax(tag_space, dim=0)

    # def _batched_cell(self, ident: torch.Tensor, lstm_seq: torch.Tensor):
    #     batch_size = ident.size(0)

    #     embeds = self.word_embeddings(ident)
    #     nn.utils.rnn.pack_padded_sequence()

    # def forward(self, unrolled_node: List[torch.Tensor]):
    #     # For now only use batches of size 0

    #     # batch_size = unrolled_node[0].size()[0]
    #     lstm_init = self.lstm_init.view(-1, 1, self.tagset_size)
    #     print(f'lstm_init: {lstm_init.size()}')
    #     # lstm_init = lstm_init.repeat(batch_size, 1, 1)
    #     # First leaf
    #     leaf_0_tag_scores = self._cell(unrolled_node[0], lstm_init)

    #     # Second leaf
    #     leaf_1_tag_scores = self._cell(unrolled_node[1], lstm_init)

    #     # Root
    #     childs_out = torch.stack([leaf_0_tag_scores, leaf_1_tag_scores])
    #     lstm_seq = torch.cat((self.lstm_init, childs_out), 0)
    #     return self._cell(unrolled_node[2], lstm_seq)

    def __init__(self):
        super(TrivialTreeTaggerUnrolled, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def unroll(node: Node) -> List[torch.Tensor]:

        return [
            torch.tensor(ident_to_id(node.childs[0]), dtype=torch.long),
            torch.tensor(ident_to_id(node.childs[1]), dtype=torch.long),
            torch.tensor(ident_to_id(node), dtype=torch.long)
        ]
