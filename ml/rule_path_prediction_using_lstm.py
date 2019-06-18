#!/usr/bin/env python3

import sys
from random import shuffle

from common import load_bag, sanitize_path
from pycore import Symbol, Rule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


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

    def forward(self, ident, remainder=None, hidden=None):
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

    def initial_hidden(self):
        return (torch.zeros([1, 1, self.hidden_dim], dtype=torch.float),
                torch.zeros([1, 1, self.hidden_dim], dtype=torch.float))

    def initial_remainder(self):
        return [torch.zeros([1, self.remainder_dim], dtype=torch.float)]


def predict(model, term):
    # Get remainders of prev
    remainders = [predict(model, child)[1] for child in term.childs]

    hidden = model.initial_hidden()

    ident_tensor = torch.tensor(
        [ident_to_ix[term.ident]], dtype=torch.long)

    if len(remainders) == 0:
        remainders = model.initial_remainder()

    for incoming_remainder in remainders:
        out, hidden, remainder = model(
            ident_tensor, incoming_remainder, hidden)

    (out, hidden, remainder) = model(ident_tensor)

    return out, remainder


def validate(model, rule_to_ix, samples):
    ranks = [0] * len(rule_to_ix)

    for sample in samples:
        log_probs, _ = predict(model, sample.initial.at(sample.path))
        probs = log_probs.exp().flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(rule_to_ix[sample.rule])
        ranks[pos] += 1

    samples_size = len(samples)
    return [rank / samples_size for rank in ranks]


def plot_ranks(ranks):
    fig = plt.figure(figsize=(8, 6))
    # Transpose ranks
    ranks = list(map(list, zip(*ranks)))
    x = range(len(ranks[0]))
    for i, rank in enumerate(ranks[:5]):
        plt.plot(x, rank, label=f'top {i+1}')
    plt.legend()
    plt.savefig('../out/ml/lstm-ranks.svg')


def plot_prediction(model, rule_to_ix, samples, ix, loss):
    ranks = validate(model, rule_to_ix, samples)
    fig = plt.figure(figsize=(8, 6))
    x = list(range(len(ranks[:10])))
    plt.bar(x, height=ranks[:10])
    plt.xticks(x, [f'top {i+1}' for i in x])
    plt.title(f'top of {len(samples)} (loss: {loss:.2f})')
    plt.savefig(f'../out/ml/lstm-hist-{ix:03}.svg')


def highlight_cell(cells, ax=None, **kwargs):
    ax = ax or plt.gca()
    for (x, y) in cells:
        rect = plt.Rectangle((x-.5, y-0.5), 1, 1, fill=False, **kwargs)
        ax.add_patch(rect)
    return rect


def plot_part_prediction(model, sample, ix_to_rule, rule_to_ix):
    rules = [f'$ {r.latex} $' for i, r in ix_to_rule.items()]
    rules_ix = [i for i, r in ix_to_rule.items()]

    with torch.no_grad():
        y_ticks = []
        prob_matrix = []
        path_to_ix = {}
        for i, (path, part) in enumerate(sample.initial.parts_with_path):
            path_to_ix[str(path)] = i
            log_probs, _ = predict(model, part)
            probs = log_probs.exp().flatten().tolist()
            prob_matrix.append(probs)
            y_ticks.append(f'$ {part.latex} $')
        plt.figure(figsize=(8, 6))
        plt.imshow(prob_matrix, cmap='Reds', interpolation='nearest')
        plt.colorbar()
        plt.xticks(rules_ix, rules, rotation=45)
        plt.yticks(list(range(len(y_ticks))), y_ticks)

        for fit in sample.fits:
            rule_ix = rule_to_ix[fit.rule.verbose]
            ix = path_to_ix[str(fit.path)]
            highlight_cell([(rule_ix, ix)], color="limegreen", linewidth=3)

        plt.savefig(
            f'../out/ml/lstm-single-prediction-{sanitize_path(str(sample.initial))}.svg')
        plt.show()


def plot_used_rules(bag):
    plt.figure(figsize=(8, 6))
    rules = bag.meta.rules
    x = range(len(rules))
    sum_of_rules = sum([rule.fits for rule in rules])
    plt.barh(
        x, width=[rule.fits/sum_of_rules for rule in rules], align='center')
    plt.yticks(x, [f'$ {stat.rule.latex} $' for stat in rules])
    plt.tight_layout()
    plt.savefig(
        f'../out/ml/lstm-rule-hist.svg')


if __name__ == "__main__":
    torch.manual_seed(1)
    bag = load_bag()
    print('Loaded')

    plot_used_rules(bag)

    # idents = list(bag.meta.used_idents)
    idents = bag.meta.idents
    ident_to_ix = {ident: i for i, ident in enumerate(idents)}

    print(f'idents: {str.join(", ", idents)}')

    rules = [stat.rule for stat in bag.meta.rules]
    rule_to_ix = {rule.verbose: i for i, rule in enumerate(rules)}
    ix_to_rule = {i: rule for i, rule in enumerate(rules)}

    # Model
    model = IdentTreeModeler(ident_size=len(idents),
                             remainder_dim=32,
                             hidden_dim=64,
                             embedding_dim=32,
                             rules_size=len(rules))

    # Data
    class Sample:
        def __init__(self, initial, rule, path):
            self.initial = initial
            self.rule = rule
            self.path = path

    samples = [Sample(sample.initial, fit.rule.verbose, fit.path)
               for sample in bag.samples for fit in sample.fits]

    shuffle(samples)

    samples_size = len(samples)
    print(f'Working on {samples_size} samples and {len(bag.meta.rules)} rules')

    test_size = samples_size // 10
    test_set = samples[:test_size]
    trainings_set = samples[test_size:]

    BATCH_SIZE = 8
    batches = [trainings_set[i:i+BATCH_SIZE]
               for i in range(0, len(trainings_set), BATCH_SIZE)]

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.002)
    MIN_VALUE = 10
    scales = [max(MIN_VALUE, v.fits)**-1.0 for v in bag.meta.rules]
    loss_function = nn.NLLLoss(weight=torch.FloatTensor(scales))
    ranks = []

    EARLY_ABORT_FACTOR = 0.001
    prev_loss = float("inf")

    for epoch in range(1):
        total_loss = 0
        for batch in batches:
            model.zero_grad()
            for sample in batch:

                log_probs, _ = predict(model, sample.initial.at(sample.path))
                log_probs = log_probs.view(1, -1)
                expected = torch.tensor(
                    [rule_to_ix[sample.rule]], dtype=torch.long)

                loss = loss_function(log_probs, expected)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            rank = validate(model, rule_to_ix, test_set)
            ranks.append(rank)
            error_pc = (1.-rank[0])*100.0
            print(f'[{epoch}] Error: {error_pc:.3}%  loss: {total_loss:.6}')

        if total_loss * (1+EARLY_ABORT_FACTOR) > prev_loss:
            print('Early abbort')
            break
        prev_loss = total_loss

    plot_ranks(ranks)
    for sample in bag.samples[:3]:
        plot_part_prediction(model, sample, ix_to_rule, rule_to_ix)
