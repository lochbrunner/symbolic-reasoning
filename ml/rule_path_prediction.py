#!/usr/bin/env python3

import sys
from random import shuffle

from pycore import Symbol, Rule
from common import load_bag, sanitize_path, ProgressBar, clear_line
from visualisations import *

from models import IdentTreeLstmModeler, RootIdentModeler, BiasedModeler
from ml_common import predict_rule, create_batches

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import argparse


@torch.no_grad()
def validate_sample(predict_rule, rule_to_ix, samples):
    '''Validates the model based on the samples.

    Returns a list [fraction of top 1, top2, ...]
    '''
    ranks = [0] * len(rule_to_ix)

    for sample in samples:
        log_probs, _ = predict_rule(sample.initial.at(sample.path))
        probs = log_probs.exp().flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(rule_to_ix[sample.rule])
        ranks[pos] += 1

    samples_size = len(samples)
    return [rank / samples_size for rank in ranks]


def create_model(name, ident_size, rules_size):
    if name == 'bias':
        return BiasedModeler(rules_size)
    elif name == 'root':
        return RootIdentModeler(ident_size, 32, rules_size)
    elif name == 'lstm':
        return IdentTreeLstmModeler(ident_size=ident_size,
                                    remainder_dim=32,
                                    hidden_dim=64,
                                    embedding_dim=32,
                                    rules_size=rules_size)
    else:
        raise f'Unknown model {name}!'


def main(model_name):
    torch.manual_seed(1)
    bag = load_bag()
    print('Loaded')

    plot_used_rules(bag)

    # Dictionaries
    idents = bag.meta.idents
    ident_to_ix = {ident: i for i, ident in enumerate(idents)}

    print(f'idents: {str.join(", ", idents)}')

    rules = [stat.rule for stat in bag.meta.rules]
    rule_to_ix = {rule.verbose: i for i, rule in enumerate(rules)}

    # Model

    model = create_model(model_name, len(idents), len(rules))

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
    batches = create_batches(trainings_set, BATCH_SIZE)

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.0002)
    MIN_VALUE = 10
    scales = [MIN_VALUE*max(MIN_VALUE, v.fits)**-1.0 for v in bag.meta.rules]
    loss_function = nn.NLLLoss(weight=torch.FloatTensor(scales))
    ranks = []

    EARLY_ABORT_FACTOR = 0.001
    prev_loss = float("inf")

    progress_bar = ProgressBar(0, len(batches))

    for epoch in range(6):
        total_loss = 0
        for i, batch in enumerate(batches):
            progress_bar.update(i)
            model.zero_grad()
            for sample in batch:
                log_probs, _ = predict_rule(model, sample.initial.at(sample.path), ident_to_ix)
                log_probs = log_probs.view(1, -1)
                expected = torch.tensor(
                    [rule_to_ix[sample.rule]], dtype=torch.long)

                loss = loss_function(log_probs, expected)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()

        # Validate
        rank = validate_sample(lambda part: predict_rule(model, part, ident_to_ix), rule_to_ix, test_set)
        ranks.append(rank)
        error_pc = (1.-rank[0])*100.0
        clear_line()
        print(f'[{epoch}] Error: {error_pc:.3}%  loss: {total_loss:.6}')

        if total_loss * (1+EARLY_ABORT_FACTOR) > prev_loss:
            print('Early abbort')
            break
        prev_loss = total_loss

    # Visualization of results
    plot_ranks(ranks)

    with torch.no_grad():
        show_part_predictions(
            lambda part: predict_rule(model, part, ident_to_ix), bag.samples, rule_to_ix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rule and path prediction.')
    parser.add_argument('-m', '--model', type=str, choices=['bias', 'root', 'lstm'],
                        default='lstm', help='The model to use')
    args = parser.parse_args()
    main(args.model)
