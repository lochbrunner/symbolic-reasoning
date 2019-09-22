#!/usr/bin/env python3

from generate import create_samples
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


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


def ident_to_id(ident):
    return ord(ident) - 97


samples, idents, tags = create_samples(size=5)

print(f'samples: {len(samples)}')
print(f'idents: {idents}')
print(f'tags: {len(tags)}')


EMBEDDING_DIM = 8
HIDDEN_DIM = 8

model = LSTMTagger(len(idents), len(tags), EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


@torch.no_grad()
def validate(samples):
    true = 0
    for tag, feature in samples:
        sequence = [ident_to_id(ident) for ident in feature]
        sequence = torch.tensor(sequence, dtype=torch.long)
        tag_scores = model(sequence)

        _, arg_max = tag_scores.max(0)
        if arg_max == tag:
            true += 1
    return float(true) / float(len(samples))


for epoch in range(4):
    epoch_loss = 0
    for tag, feature in samples:
        model.zero_grad()
        sequence = [ident_to_id(ident) for ident in feature]
        sequence = torch.tensor(sequence, dtype=torch.long)
        tag_scores = model(sequence).view(1, -1)

        tag = torch.tensor([tag], dtype=torch.long)

        loss = loss_function(tag_scores, tag)
        epoch_loss += loss.item() / len(samples)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        error = 1.-validate(samples)
        print(f'[{epoch}] error: {error} loss: {epoch_loss}')
