#!/usr/bin/env python3

import sys

from common import load_trace, Step
from pycore import Trace, Symbol, Rule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

HIDDEN_DIM = 64
REMAINDER_DIM = 32


class IdentTreeModeler(nn.Module):
    """
    Based on the subtree
    """

    def __init__(self, ident_size, remainder_dim, hidden_dim, embedding_dim, rules_size):
        super(IdentTreeModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.out_dim = rules_size
        self.remainder_dim = remainder_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(input_size=embedding_dim +
                            remainder_dim, hidden_size=self.hidden_dim)
        self.linear_out = nn.Linear(embedding_dim, rules_size)

    def forward(self, ident, remainder=None, hidden=None, ):
        if remainder is None:
            remainder = torch.zeros([1, self.remainder_dim], dtype=torch.float)

        if hidden is None:
            hidden = (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float),
                      torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))

        embeds = self.embeddings(ident).view((1, -1))

        inputs = torch.cat((embeds, remainder), 1)

        inputs = inputs.view(len(inputs), 1, -1)
        out, hidden = self.lstm(inputs, hidden)
        (out, remainder) = torch.split(
            out, split_size_or_sections=self.embedding_dim, dim=2)

        out = F.relu(self.linear_out(out))
        out = F.log_softmax(out, dim=2)

        return out, hidden, remainder.view(1, -1)


def predict(model, term):

    hidden = (torch.zeros([1, 1, HIDDEN_DIM], dtype=torch.float),
              torch.zeros([1, 1, HIDDEN_DIM], dtype=torch.float))

    ident_tensor = torch.tensor(
        [ident_to_ix[term.ident]], dtype=torch.long)

    # Get remainders of prev
    remainders = [predict(model, child)[1] for child in term.childs]

    if len(remainders) == 0:
        remainders = [torch.zeros([1, REMAINDER_DIM], dtype=torch.float)]

    for incoming_remainder in remainders:
        out, hidden, remainder = model(
            ident_tensor, incoming_remainder, hidden)

    (out, hidden, remainder) = model(ident_tensor)

    return out, remainder


def plot_prediction(model, rule_to_ix, steps, ix, loss):

    ranks = [0] * len(rule_to_ix)

    for step in steps:
        log_probs, _ = predict(model, step.initial)
        probs = log_probs.exp().flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(rule_to_ix[step.rule])
        ranks[pos] += 1

    fig = plt.figure()
    x = list(range(len(ranks[:10])))
    plt.bar(x, height=ranks[:10])
    plt.xticks(x, [f'top {i+1}' for i in x])
    plt.title(f'top of {len(steps)} (loss: {loss:.2f})')
    plt.savefig(f'../out/ml/lstm-hist-{ix:03}.svg')


def traverse(term):
    for child in term.childs:
        traverse(child)
    print(term.ident)


if __name__ == "__main__":
    trace = load_trace()
    torch.manual_seed(1)

    idents = list(trace.meta.used_idents)
    ident_to_ix = {ident: i for i, ident in enumerate(idents)}

    print(f'idents: {str.join(", ", idents)}')

    rules = [str(rule.reverse) for rule in trace.meta.rules]
    # Remove duplicates
    rules = list(set(rules))
    rule_to_ix = {rule: i for i, rule in enumerate(rules)}

    # Model
    model = IdentTreeModeler(ident_size=len(idents),
                             remainder_dim=REMAINDER_DIM,
                             hidden_dim=HIDDEN_DIM,
                             embedding_dim=32,
                             rules_size=len(rules))

    steps = [Step(initial=step.initial,
                  deduced=str(step.deduced),
                  rule=str(step.rule.reverse),
                  path=step.path)
             for step in trace.all_steps]

    with torch.no_grad():
        out, _ = predict(model, steps[0].initial)
        probs = out.exp().flatten().tolist()
        print(f'Probabilities: {probs}')

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    for epoch in range(101):
        total_loss = 0
        for step in steps:
            model.zero_grad()

            log_probs, _ = predict(model, step.initial)
            log_probs = log_probs.view(1, -1)
            expected = torch.tensor([rule_to_ix[step.rule]], dtype=torch.long)

            loss = loss_function(log_probs, expected)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'loss[{epoch}]: {total_loss}')
        if epoch % 5 == 0:
            with torch.no_grad():
                plot_prediction(model, rule_to_ix, steps, epoch, total_loss)

    with torch.no_grad():
        out, _ = predict(model, steps[0].initial)
        probs = out.exp().flatten().tolist()
        print(f'Probabilities: {probs}')
