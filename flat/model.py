import torch
import torch.nn as nn
import torch.nn.functional as F

from .lstm import LSTMCellOwn, LSTMCellOptimized, LSTMCellOptimizedTwo, LSTMCellRebuilt


class LSTMTagger(nn.Module):
    '''From https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch'''

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores[-1, :]


class LSTMTaggerOwn(nn.Module):
    '''Using Own LSTM implementations'''

    @staticmethod
    def choices():
        return ['rebuilt', 'own', 'optimized', 'optimized-two']

    @staticmethod
    def contains_implementation(name):
        return name in LSTMTaggerOwn.choices()

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, implementation='torch'):
        super(LSTMTaggerOwn, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if implementation == 'optimized':
            self.lstm_cell = LSTMCellOptimized(embedding_dim, hidden_dim)
        elif implementation == 'optimized-two':
            self.lstm_cell = LSTMCellOptimized(embedding_dim, hidden_dim)
        elif implementation == 'rebuilt':
            self.lstm_cell = LSTMCellRebuilt(embedding_dim, hidden_dim)
        elif implementation == 'own':
            self.lstm_cell = LSTMCellOwn(embedding_dim, hidden_dim)
        else:
            raise Exception(f'Unknown implementation: \'{implementation}\'')
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.embedding_dim = embedding_dim

    def forward(self, sequence):
        hidden = torch.zeros(self.embedding_dim)
        cell = torch.zeros(self.embedding_dim)
        for word in sequence:
            embeds = self.word_embeddings(word)
            hidden, cell = self.lstm_cell(embeds.view(-1), hidden, cell)

        tag_space = self.hidden2tag(hidden.view(-1))
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores


class LSTTaggerBuiltinCell(nn.Module):

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTTaggerBuiltinCell, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.embedding_dim = embedding_dim

    def forward(self, sequence):
        hidden = torch.zeros(self.embedding_dim)
        cell = torch.zeros(self.embedding_dim)
        for word in sequence:
            embeds = self.word_embeddings(word)
            hidden, cell = self.lstm_cell(
                embeds.view(1, -1), (hidden.view(1, -1), cell.view(1, -1)))

        tag_space = self.hidden2tag(hidden.view(-1))
        tag_scores = F.log_softmax(tag_space, dim=0)
        return tag_scores
