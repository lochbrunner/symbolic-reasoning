#!/usr/bin/env python3

import sys
from random import shuffle

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
        lstm_out, hidden = self.lstm(inputs, hidden)
        (out, remainder) = torch.split(
            lstm_out, split_size_or_sections=self.embedding_dim, dim=2)

        out = F.relu(self.linear_out(out))
        out = F.log_softmax(out, dim=2)

        return out, hidden, remainder.view(1, -1)


def predict(model, term):
    # Get remainders of prev
    remainders = [predict(model, child)[1] for child in term.childs]

    hidden = (torch.zeros([1, 1, HIDDEN_DIM], dtype=torch.float),
              torch.zeros([1, 1, HIDDEN_DIM], dtype=torch.float))

    ident_tensor = torch.tensor(
        [ident_to_ix[term.ident]], dtype=torch.long)

    if len(remainders) == 0:
        remainders = [torch.zeros([1, REMAINDER_DIM], dtype=torch.float)]

    for incoming_remainder in remainders:
        out, hidden, remainder = model(
            ident_tensor, incoming_remainder, hidden)

    (out, hidden, remainder) = model(ident_tensor)

    return out, remainder


def validate(model, rule_to_ix, steps):
    ranks = [0] * len(rule_to_ix)

    for step in steps:
        log_probs, _ = predict(model, step.initial.get(step.path))
        probs = log_probs.exp().flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(rule_to_ix[step.rule])
        ranks[pos] += 1

    steps_size = len(steps)
    return [rank / steps_size for rank in ranks]


def plot_ranks(ranks):
    fig = plt.figure(figsize=(8, 6))
    # Transpose ranks
    ranks = list(map(list, zip(*ranks)))
    x = range(len(ranks[0]))
    for i, rank in enumerate(ranks[:5]):
        plt.plot(x, rank, label=f'top {i+1}')
    plt.legend()
    plt.savefig('../out/ml/lstm-ranks.svg')


def plot_prediction(model, rule_to_ix, steps, ix, loss):
    ranks = validate(model, rule_to_ix, steps)
    fig = plt.figure(figsize=(8, 6))
    x = list(range(len(ranks[:10])))
    plt.bar(x, height=ranks[:10])
    plt.xticks(x, [f'top {i+1}' for i in x])
    plt.title(f'top of {len(steps)} (loss: {loss:.2f})')
    plt.savefig(f'../out/ml/lstm-hist-{ix:03}.svg')


def plot_part_prediction(model, step, ix_to_rule):
    # TODO: use LaTeX formated rules
    rules = [r for i, r in ix_to_rule.items()]
    rules_ix = [i for i, r in ix_to_rule.items()]

    with torch.no_grad():
        y_ticks = []
        prob_matrix = []
        for part in step.initial.parts:

            log_probs, _ = predict(model, part)
            probs = log_probs.exp().flatten().tolist()
            prob_matrix.append(probs)
            y_ticks.append(str(part))
        plt.figure(figsize=(8, 6))
        plt.imshow(prob_matrix, cmap='Reds', interpolation='nearest')
        plt.colorbar()
        plt.xticks(rules_ix, rules, rotation=45)
        plt.yticks(list(range(len(y_ticks))), y_ticks)
        plt.savefig(
            f'../out/ml/lstm-single-prediction-{str(step.initial)}.svg')
        plt.show()


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
    ix_to_rule = {i: rule for i, rule in enumerate(rules)}

    # Model
    model = IdentTreeModeler(ident_size=len(idents),
                             remainder_dim=REMAINDER_DIM,
                             hidden_dim=HIDDEN_DIM,
                             embedding_dim=32,
                             rules_size=len(rules))

    # Data
    steps = [Step(initial=step.deduced,
                  deduced=str(step.initial),
                  rule=str(step.rule.reverse),
                  path=step.path)
             for step in trace.all_steps]

    shuffle(steps)

    steps_size = len(steps)
    print(f'Working on {steps_size} steps')

    test_size = steps_size // 10
    test_set = steps[:test_size]
    trainings_set = steps[test_size:]

    BATCH_SIZE = 8
    batches = [trainings_set[i:i+BATCH_SIZE]
               for i in range(0, len(trainings_set), BATCH_SIZE)]

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()
    ranks = []

    EARLY_ABORT_FACTOR = 0.001
    prev_loss = float("inf")

    for epoch in range(6):
        total_loss = 0
        for batch in batches:
            model.zero_grad()
            for step in batch:

                log_probs, _ = predict(model, step.initial.get(step.path))
                log_probs = log_probs.view(1, -1)
                expected = torch.tensor(
                    [rule_to_ix[step.rule]], dtype=torch.long)

                loss = loss_function(log_probs, expected)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            rank = validate(model, rule_to_ix, test_set)
            ranks.append(rank)
            error_pc = (1.-rank[0])*100.0
            print(f'[{epoch}] Error: {error_pc:.3}%  loss: {total_loss}')

        if total_loss * (1+EARLY_ABORT_FACTOR) > prev_loss:
            print('Early abbort')
            break
        prev_loss = total_loss

    plot_ranks(ranks)
    for step in steps[:10]:
        plot_part_prediction(model, step, ix_to_rule)
