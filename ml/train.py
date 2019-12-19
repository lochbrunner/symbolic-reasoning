#!/usr/bin/env python3

import logging
import argparse
from functools import reduce
import operator
import signal
import sys

import numpy as np

import torch
import torch.optim as optim
from torch.utils import data
from torch import nn

from dataset import create_scenario, scenarios_choices, ScenarioParameter
from dataset.transformers import Embedder
from models import create_model, all_models

from common.timer import Timer
from common.utils import printProgressBar, clearProgressBar
from common.parameter_search import LearningParmeter
from common import io


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, load_model: str = None, save_model: str = None, device: str = 'auto'):
        self.report_rate = report_rate
        self.load_model = load_model
        self.save_model = save_model
        self.device = device


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: data.DataLoader):
    true = 0
    false = 0
    # We assume batchsize of 1
    assert dataloader.batch_size == 1
    for x, y, s in dataloader:
        x = model(x, s)
        x = x.squeeze()
        x = x.cpu().numpy()
        predict = np.argmax(x, axis=0)

        y = y.squeeze()
        truth = y.cpu().numpy()

        true += np.sum(predict == truth)
        false += np.sum(predict != truth)
    return float(true) / float(true + false)


def main(exe_params: ExecutionParameter, learn_params: LearningParmeter, scenario_params: ScenarioParameter):
    device = torch.device(exe_params.device)
    logging.info(f'Using device: {device}')

    # Creating dataset and model
    if exe_params.load_model:
        dataset, model, optimizer, _ = io.load(exe_params.exe_params.load_model, device)
    else:
        pad_token = 0
        timer = Timer('Creating fresh workspace')
        dataset = create_scenario(params=scenario_params, device=device, pad_token=pad_token)
        model = create_model(learn_params.model_name,
                             vocab_size=dataset.vocab_size,
                             tagset_size=dataset.tag_size,
                             pad_token=pad_token,
                             blueprint=Embedder.blueprint(scenario_params),
                             hyper_parameter=learn_params.model_hyper_parameter)
        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=learn_params.learning_rate)
        optimizer = optim.Adadelta(model.parameters())
        timer.stop_and_log()

    model.train()

    # Loading data
    train_loader_params = {'batch_size': learn_params.batch_size,
                           'shuffle': True,
                           'num_workers': 0}

    validate_loader_params = {'batch_size': 1,
                              'shuffle': False,
                              'num_workers': 0}
    training_dataloader = data.DataLoader(dataset, **train_loader_params)
    validation_dataloader = data.DataLoader(dataset, **validate_loader_params)

    weight = torch.as_tensor(dataset.label_weight, device=device, dtype=torch.float)
    loss_function = nn.NLLLoss(reduction='mean', weight=weight)

    # Training
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def early_abort(_signal, _frame):
        clearProgressBar()
        logging.warning('Early abort')
        io.save(exe_params.save_model, model, optimizer, scenario_params, learn_params)
        sys.exit(1)
    signal.signal(signal.SIGINT, early_abort)

    timer = Timer(f'Training per sample:')
    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        model.zero_grad()
        for x, y, s in training_dataloader:
            optimizer.zero_grad()
            x = model(x, s)
            # batch x tags
            loss = loss_function(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()
            epoch_loss += loss
        if (epoch+1) % exe_params.report_rate == 0:
            timer.pause()
            error = validate(model, validation_dataloader)
            clearProgressBar()
            error = (1. - error) * 100.
            loss = learn_params.batch_size * epoch_loss
            logging.info(f'#{epoch} Loss: {loss:.3f}  Error: {error:.1f}%')
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs)
    clearProgressBar()
    timer.stop_and_log_average(learn_params.num_epochs*len(dataset))

    signal.signal(signal.SIGINT, original_sigint_handler)

    io.save(exe_params.save_model, model, optimizer, scenario_params, learn_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # Execution parameter
    parser.add_argument('-i', '--load-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-o', '--save-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-u', '--update-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-r', '--report-rate', type=int, default=20)
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda', 'auto'], default='auto')

    # Learning parameter
    parser.add_argument('-n', '--num-epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-c', '--gradient-clipping', type=float, default=0.1)
    parser.add_argument('-m', '--model', choices=[m for m in all_models], default='LstmTreeTagger')

    # Scenario
    parser.add_argument('-s', '--scenario', type=str,
                        default='permutation', choices=scenarios_choices())
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of the used nodes.')
    parser.add_argument('--pattern-depth', type=int, default=1,
                        help='The depth of the pattern nodes.')
    parser.add_argument('--spread', type=int, default=2)
    parser.add_argument('--max-size', type=int, default=120)
    parser.add_argument('--num-labels', type=int, default=2)

    args = parser.parse_args()
    loglevel = 'INFO' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging._nameToLevel[loglevel],
        format='%(message)s'
    )

    def get_device(device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            return device

    exec_params = ExecutionParameter(
        report_rate=args.report_rate, load_model=args.load_model or args.update_model,
        save_model=args.save_model or args.update_model,
        device=get_device(args.device))

    learn_params = LearningParmeter(
        model_name=args.model,
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, gradient_clipping=args.gradient_clipping,
        model_hyper_parameter={})

    scenario_params = ScenarioParameter(
        scenario=args.scenario, depth=args.depth, spread=args.spread,
        max_size=args.max_size,
        pattern_depth=args.pattern_depth,
        num_labels=args.num_labels)

    main(exec_params, learn_params, scenario_params=scenario_params)
