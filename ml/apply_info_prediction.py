#!/usr/bin/env python3

import sys
from pycore import Trace, Symbol, Rule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


def load_trace():
    file = '../out/generator/trace.bin'

    try:
        trace = Trace.load(file)
    except Exception as e:
        print(f'Error loading {file}: {e}')
        sys.exit(1)
    return trace


class Step:
    def __init__(self, initial, deduced, rule, path):
        self.initial = initial
        self.deduced = deduced
        self.rule = rule
        self.path = path


class IdentModeler(nn.Module):
    """
    Using fully connected layer
    """

    def __init__(self, ident_size, embedding_dim, rules_size):
        super(IdentModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, rules_size)

    def forward(self, prev_symbol):
        """
        Predicts the rule to apply
        """
        embeds = self.embeddings(prev_symbol).view((1, -1))
        h = F.relu(self.linear1(embeds))
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)


def print_prediction(pre_symbol_to_ix, ix_to_rule, model, steps):
    with torch.no_grad():
        GREEN = '\033[92m'
        ENDC = '\033[0m'
        for step in steps:
            symbol_tensor = torch.tensor(
                [pre_symbol_to_ix[step.initial]], dtype=torch.long)

            log_probs = model(symbol_tensor).exp().flatten().tolist()

            # Find tops
            tops = sorted([(v, i)
                           for i, v in enumerate(log_probs)], reverse=True)[:5]

            print(f'prev: {step.initial} -> {step.deduced}')
            print(f'expected: {step.rule} @{step.path}')

            for (v, i) in tops:
                r = ix_to_rule[i]
                if r == step.rule:
                    print(f'{GREEN}{r} with {v}{ENDC}')
                else:
                    print(f'{r} with {v}')
            print('')


def plot_tops(rule_to_ix, pre_symbol_to_ix, model, steps):
    ranks = [0] * len(pre_symbol_to_ix)

    for step in steps:
        symbol_tensor = torch.tensor(
            [pre_symbol_to_ix[step.initial]], dtype=torch.long)

        log_probs = model(symbol_tensor).flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(log_probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(rule_to_ix[step.rule])

        ranks[pos] += 1

    fig = plt.figure()
    x = list(range(len(ranks[:10])))
    plt.bar(x, height=ranks[:10])
    plt.xticks(x, [f'top {i+1}' for i in x])
    plt.title(f'top of {len(steps)}')
    plt.savefig('../out/ml/hist.svg')
    # plt.show()


if __name__ == "__main__":

    trace = load_trace()

    torch.manual_seed(1)

    steps = [Step(initial=str(step.deduced), deduced=str(step.initial), rule=str(step.rule.reverse), path=list(step.path))
             for step in trace.all_steps]

    used_symbols = set([step.initial for step in steps])
    pre_symbol_to_ix = {s: i for i, s in enumerate(used_symbols)}

    # Used later for LSTM
    # idents = list(trace.meta.used_idents)
    # ident_to_ix = {ident: i for i, ident in enumerate(idents)}

    rules = [str(rule.reverse) for rule in trace.meta.rules]
    rule_to_ix = {rule: i for i, rule in enumerate(rules)}
    ix_to_rule = {i: rule for i, rule in enumerate(rules)}

    # Model
    model = IdentModeler(len(used_symbols), 32, len(rules))

    # Trained predictions
    print_prediction(pre_symbol_to_ix, ix_to_rule, model, steps[:4])
    plot_tops(rule_to_ix, pre_symbol_to_ix, model, steps)
    print('')

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    losses = []
    for epoch in range(20):
        total_loss = 0
        for step in steps:
            symbol = torch.tensor(
                [pre_symbol_to_ix[step.initial]], dtype=torch.long)

            model.zero_grad()

            log_probs = model(symbol)

            # print(torch.tensor([rule_to_ix[step.rule]]))
            # sys.exit(0)

            loss = loss_function(log_probs, torch.tensor(
                [rule_to_ix[step.rule]], dtype=torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)

    print(losses)

    # Trained predictions
    print_prediction(pre_symbol_to_ix, ix_to_rule, model, steps[:4])

    # Plot statistics
    plot_tops(rule_to_ix, pre_symbol_to_ix, model, steps)
