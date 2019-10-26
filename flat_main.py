#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

from common.utils import printProgressBar, clearProgressBar, create_batches
from common.reports import plot_train_progess, TrainingProgress
from flat.generate import create_samples, choices as strategy_choices
from flat.model import LSTMTagger, LSTMTaggerOwn, LSTTaggerBuiltinCell

import torch
import torch.nn as nn
import torch.optim as optim

from functools import reduce
from math import isnan
import operator
import argparse
import argcomplete


torch.manual_seed(1)


def ident_to_id(ident):
    return ord(ident) - 97


@torch.no_grad()
def validate(model, samples):
    true = 0
    for tag, feature in samples:
        sequence = [ident_to_id(ident) for ident in feature]
        sequence = torch.tensor(sequence, dtype=torch.long)
        tag_scores = model(sequence)

        _, arg_max = tag_scores.max(0)
        if arg_max == tag:
            true += 1
    return float(true) / float(len(samples))


def main(strategy, num_epochs, batch_size=10, use=None, verbose=False):

    samples, idents, tags = create_samples(strategy=strategy)

    if verbose:
        print(f'samples: {len(samples)}  tags: {len(tags)}')
        print(f'idents: {idents}')

    EMBEDDING_DIM = 8
    HIDDEN_DIM = 8

    if use == None or use == 'torch':
        model = LSTMTagger(len(idents), len(tags), EMBEDDING_DIM, HIDDEN_DIM)
    elif LSTMTaggerOwn.contains_implementation(use):
        model = LSTMTaggerOwn(len(idents), len(
            tags), EMBEDDING_DIM, HIDDEN_DIM, use)
    elif use == 'torch-cell':
        model = LSTTaggerBuiltinCell(len(idents), len(
            tags), EMBEDDING_DIM, HIDDEN_DIM)

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    if verbose:
        print(f'Number of parameters: {num_parameters}')
        print(f'Scenario: {strategy} using {use}')

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, )

    progress = []
    batches = create_batches(samples, batch_size)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in batches:
            model.zero_grad()
            for tag, feature in batch:
                sequence = [ident_to_id(ident) for ident in feature]
                sequence = torch.tensor(sequence, dtype=torch.long)
                tag_scores = model(sequence).view(1, -1)

                tag = torch.tensor([tag], dtype=torch.long)

                loss = loss_function(tag_scores, tag)
                if isnan(loss):
                    print('Invalid loss detected! Aborting...')
                    clearProgressBar()
                    return

                epoch_loss += loss.item() / len(batch)
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
        if epoch % 10 == 0:
            error = 1.-validate(model, samples)
            progress.append(TrainingProgress(epoch, epoch_loss, error))
            if verbose:
                clearProgressBar()
                print(f'#{epoch} Loss: {epoch_loss}  Error: {error}')
        printProgressBar(epoch, num_epochs)

    clearProgressBar()
    plot_train_progess(progress, strategy, use,
                       dump_filename='./reports/flat/dump.p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('flat training')
    parser.add_argument('-s', '--strategy', type=str,
                        default='permutation', choices=strategy_choices() + ['all'])
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=5)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument(
        '--use', choices=['torch', 'torch-cell'] + LSTMTaggerOwn.choices(), nargs='*', default=['torch'])

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.strategy == 'all':
        for strategy in strategy_choices():
            print(f'Processing: {strategy} ...')
            for use in args.use:
                main(strategy, args.num_epochs, use=use, verbose=args.verbose)
    else:
        for use in args.use:
            main(args.strategy, args.num_epochs, use=use, verbose=args.verbose)
