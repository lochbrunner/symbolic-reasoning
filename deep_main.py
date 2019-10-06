#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import torch
from torch import nn

from functools import reduce
import operator
import argparse
import argcomplete

from deep.generate import create_samples
from deep.node import Node
from common.utils import printProgressBar, clearProgressBar, create_batches
from common.reports import plot_train_progess, TrainingProgress

from typing import List, Set, Dict, Tuple, Optional

from deep.model import TreeTagger, TrivialTreeTagger

import torch
import torch.optim as optim


def ident_to_id(node: Node):
    return ord(node.ident) - 97


@torch.no_grad()
def validate(model: torch.nn.Module, samples: List[Tuple[int, Node]]):
    true = 0
    for tag, feature in samples:
        tag_scores = model(feature)

        _, arg_max = tag_scores.max(0)
        if arg_max == tag:
            true += 1
    return float(true) / float(len(samples))


def main(depth: int, spread: int, num_epochs: int, batch_size: int = 10, report_rate: int = 10):

    samples, idents, tags = create_samples(
        depth=depth, spread=spread)

    print(f'samples: {len(samples)}  tags: {len(tags)} idents: {len(idents)}')
    # print('\n\n'.join([str(sample[1]) for sample in samples[:1]]))

    model = TrivialTreeTagger(len(idents), len(tags),
                              embedding_size=32, hidden_size=len(tags))

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    print(f'Number of parameters: {num_parameters}')

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    progress: List[TrainingProgress] = []
    batches = create_batches(samples, batch_size)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in batches:
            model.zero_grad()
            for tag, node in batch:
                tag_scores = model(node)
                tag = torch.tensor([tag], dtype=torch.long)
                loss = loss_function(tag_scores.view(1, -1), tag)
                epoch_loss += loss.item() / len(batch)
                loss.backward()
            optimizer.step()

        if epoch % report_rate == 0:
            error = 1.-validate(model, samples)
            progress.append(TrainingProgress(epoch, epoch_loss, error))
        printProgressBar(epoch, num_epochs)

    clearProgressBar()
    plot_train_progess(progress, strategy='permutation', use='simple',
                       plot_filename='./reports/deep/training.{}.{}.svg',
                       dump_filename='./reports/deep_dump.p')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('deep training')

    parser.add_argument('-d', '--depth', type=int, default=2,
                        help='The depth of the used nodes.')
    parser.add_argument('-s', '--spread', type=int, default=2)
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=10)
    parser.add_argument('-r', '--report-rate', type=int, default=10)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    main(depth=args.depth,
         spread=args.spread,
         num_epochs=args.num_epochs,
         batch_size=args.batch_size,
         report_rate=args.report_rate
         )
