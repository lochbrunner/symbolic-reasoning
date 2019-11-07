#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import torch
from torch import nn

from functools import reduce
import operator
import argparse
import argcomplete
from math import isnan
from typing import List, Set, Dict, Tuple, Optional

from deep.generate import create_samples_permutation, scenarios_choices
from deep.node import Node
from common.utils import printProgressBar, clearProgressBar, create_batches
from common.reports import plot_train_progess, TrainingProgress
from common.parameter_search import LearningParmeter


from deep.model import TreeTagger, TrivialTreeTagger

import torch
import torch.optim as optim


@torch.no_grad()
def validate(model: torch.nn.Module, samples: List[Tuple[int, Node]]):
    true = 0
    for tag, feature in samples:
        tag_scores = model(feature)

        _, arg_max = tag_scores.max(0)
        if arg_max == tag:
            true += 1
    return float(true) / float(len(samples))


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, verbose: bool = True,
                 load_model: str = None, save_model: str = None):
        self.report_rate = report_rate
        self.verbose = verbose
        self.load_model = load_model
        self.save_model = save_model


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread


def main(exe_params: ExecutionParameter, learn_params: LearningParmeter,
         scenario_params: ScenarioParameter):

    samples, idents, tags = create_samples_permutation(
        depth=scenario_params.depth, spread=scenario_params.spread)

    if exe_params.verbose:
        print(
            f'samples: {len(samples)}  tags: {len(tags)} idents: {len(idents)}')
        print('First sample looks')
        print(samples[0][1])

    model = TrivialTreeTagger(vocab_size=len(idents), tagset_size=len(tags),
                              hyper_parameter=learn_params.model_hyper_parameter)

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    if exe_params.verbose:
        print(f'Number of parameters: {num_parameters}')

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if exe_params.load_model is not None:
        print(f'Loading model from {exe_params.load_model} ...')
        checkpoint = torch.load(exe_params.load_model)
        file_use = checkpoint['use']
        current_use = 'default'
        if file_use != current_use:
            raise Exception(
                f'Loaded model contains {file_use} but {current_use} is specified for this training.')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()

    progress: List[TrainingProgress] = []
    batches = create_batches(samples, learn_params.batch_size)
    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        for batch in batches:
            model.zero_grad()
            for tag, node in batch:
                tag_scores = model(node)
                tag = torch.tensor([tag], dtype=torch.long)
                loss = loss_function(tag_scores.view(1, -1), tag)
                if isnan(loss):
                    print('Invalid loss detected! Aborting...')
                    clearProgressBar()
                    return
                epoch_loss += loss.item() / len(batch)
                loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()

        if epoch % exe_params.report_rate == 0:
            error = 1.-validate(model, samples)
            progress.append(TrainingProgress(epoch, epoch_loss, error))
            if exe_params.verbose:
                clearProgressBar()
                print(f'#{epoch} Loss: {epoch_loss}  Error: {error}')
        printProgressBar(epoch, learn_params.num_epochs)

    clearProgressBar()
    plot_train_progess(progress, strategy='permutation', use='simple',
                       plot_filename='./reports/deep/training.{}.{}.svg',
                       dump_filename='./reports/deep_dump.p')

    if exe_params.save_model is not None:
        print(f'Saving model to {exe_params.save_model} ...')
        torch.save({
            'epoch': learn_params.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'use': 'default',
            'hyper_parameter': learn_params.model_hyper_parameter}, exe_params.save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('deep training')

    parser.add_argument('-s', '--scenario', type=str,
                        default='permutation', choices=scenarios_choices() + ['all'])
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of the used nodes.')
    parser.add_argument('--spread', type=int, default=2)
    parser.add_argument('-n', '--num-epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=10)
    parser.add_argument('-r', '--report-rate', type=int, default=10)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-c', '--gradient-clipping', type=float, default=0.1)
    parser.add_argument('-i', '--load-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-o', '--save-model', type=str,
                        default=None, help='Path to the model')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    exec_params = ExecutionParameter(
        report_rate=args.report_rate, verbose=args.verbose, load_model=args.load_model,
        save_model=args.save_model)

    learn_params = LearningParmeter(
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, gradient_clipping=args.gradient_clipping,
        model_hyper_parameter={'embedding_size': 32})

    scenario_params = ScenarioParameter(
        scenario=args.scenario, depth=args.depth, spread=args.spread)

    if args.scenario == 'all':
        for scenario in scenarios_choices():
            print(f'Processing: {scenario} ...')
            scenario_params.scenario = scenario
            main(exec_params, learn_params, scenario_params)
    else:
        main(exec_params, learn_params, scenario_params)
