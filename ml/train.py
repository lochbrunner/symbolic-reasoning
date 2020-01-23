#!/usr/bin/env python3

import logging
import argparse
import signal
import sys

import numpy as np

import torch
import torch.optim as optim
from torch.utils import data
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import create_scenario, scenarios_choices, ScenarioParameter
from dataset.transformers import Embedder
from models import create_model, all_models

from common.timer import Timer
from common.utils import printProgressBar, clearProgressBar
from common.parameter_search import LearningParmeter
from common import io


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, update_model: str = None, load_model: str = None,
                 save_model: str = None, device: str = 'auto', tensorboard: bool = False, **kwargs):
        self.report_rate = report_rate
        self.load_model = load_model or update_model
        self.save_model = save_model or update_model
        self.device = device
        self.tensorboard = tensorboard


class Ratio:
    def __init__(self):
        self.true = 0
        self.false = 0

    def __float__(self):
        return float(self.true) / max(1, float(self.true + self.false))

    def __str__(self):
        v = (1. - float(self)) * 100.
        return f'{v:.1f}%'

    def update(self, mask, predict, truth):
        p = predict[mask]
        t = truth[mask]

        self.true += np.sum(p == t)
        self.false += np.sum(p != t)


class Error:
    def __init__(self, with_padding=None, wo_padding=None):
        self.with_padding = with_padding or Ratio()
        self.wo_padding = wo_padding or Ratio()


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: data.DataLoader):
    error = Error()

    for x, y, m in dataloader:
        x = x.to(model.device)
        y = y.to(model.device)
        m = m.to(model.device)
        # Dimensions
        # x: batch * label * length
        # y: batch * length
        x = model(x)
        assert x.size(0) == y.size(0), f'{x.size(0)} == {y.size(0)}'
        batch_size = x.size(0)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        m = m.cpu().numpy()
        for i in range(batch_size):
            predict = np.argmax(x[i, :, :], axis=0)
            truth = y[i, :]

            error.with_padding.update((truth > -1), predict, truth)
            error.wo_padding.update((truth > 0), predict, truth)

    return error


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
                             spread=dataset.max_spread,
                             depth=dataset.max_depth,
                             hyper_parameter=learn_params.model_hyper_parameter)
        model.to(device)

        optimizer = optim.Adadelta(model.parameters())
        timer.stop_and_log()

    model.train()

    # Loading data
    train_loader_params = {'batch_size': learn_params.batch_size,
                           'shuffle': True,
                           'num_workers': 0}

    validate_loader_params = {'batch_size': 8,
                              'shuffle': False,
                              'num_workers': 0}
    training_dataloader = data.DataLoader(dataset, **train_loader_params)
    validation_dataloader = data.DataLoader(dataset, **validate_loader_params)

    weight = torch.as_tensor(dataset.label_weight, device=device, dtype=torch.float)
    loss_function = nn.NLLLoss(reduction='mean', weight=weight, ignore_index=-1)

    # Training
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def save_snapshot():
        io.save(exe_params.save_model, model, optimizer, scenario_params, learn_params, dataset)

    def early_abort(*args):
        clearProgressBar()
        logging.warning('Early abort')
        save_snapshot()
        sys.exit(1)
    signal.signal(signal.SIGINT, early_abort)

    if exe_params.tensorboard:
        writer = SummaryWriter()
        x, s, _ = next(iter(validation_dataloader))
        writer.add_graph(model, (x, s))
        writer.close()

    timer = Timer(f'Training per sample:')
    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        model.zero_grad()
        for x, y, m in training_dataloader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            optimizer.zero_grad()
            x = model(x, m)
            # batch x tags
            loss = loss_function(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()
            epoch_loss += loss
        if epoch % exe_params.report_rate == 0:
            timer.pause()
            error = validate(model, validation_dataloader)
            clearProgressBar()
            loss = learn_params.batch_size * epoch_loss
            logging.info(f'#{epoch} Loss: {loss:.3f}  Error: {error.with_padding} (wo padding: {error.wo_padding})')
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs)
    clearProgressBar()
    timer.stop_and_log_average(learn_params.num_epochs*len(dataset))

    signal.signal(signal.SIGINT, original_sigint_handler)

    save_snapshot()


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
    parser.add_argument('--tensorboard', action='store_true', default=False)

    # Learning parameter
    parser.add_argument('-n', '--num-epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-c', '--gradient-clipping', type=float, default=0.1)
    parser.add_argument('-m', '--model', choices=all_models,
                        default='FullyConnectedSegmenter', dest='model_name')

    # Scenario
    parser.add_argument('-s', '--scenario', type=str,
                        default='pattern', choices=scenarios_choices())
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of the used nodes.')
    parser.add_argument('--pattern-depth', type=int, default=1,
                        help='The depth of the pattern nodes.')
    parser.add_argument('--spread', type=int, default=2)
    parser.add_argument('--max-size', type=int, default=120)
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bag-filename', type=str, default=None, dest='filename')

    args = parser.parse_args()
    loglevel = 'INFO' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If there might be conflicts in argument names use https://stackoverflow.com/a/18677482/6863221
    main(
        exe_params=ExecutionParameter(**vars(args)),
        learn_params=LearningParmeter(**vars(args)),
        scenario_params=ScenarioParameter(**vars(args))
    )
