#!/usr/bin/env python3

import logging
import signal
import sys
import yaml


import torch
from torch import nn
import torch.optim as optim
from torch.utils import data
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

from dataset import create_scenario, scenarios_choices, ScenarioParameter
from models import create_model, all_models

from common.timer import Timer
from common.config_and_arg_parser import Parser as ArgumentParser
from common.terminal_utils import printProgressBar, clearProgressBar
from common.parameter_search import LearningParmeter
from common import io
from common.validation import validate
from common import grid_search


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, update_model: str = None, load_model: str = None,
                 save_model: str = None, device: str = 'auto', tensorboard: bool = False, statistics: str = None, **kwargs):
        self.report_rate = report_rate
        self.load_model = load_model or update_model
        self.save_model = save_model or update_model
        self.device = device
        self.tensorboard = tensorboard
        self.statistics = statistics


def dump_statistics(params: ExecutionParameter, logbooks):
    '''Dumps statistics '''
    stat = [{'parameter': hp,
             'results': [{'error': error.as_dict(), 'epoch': epoch} for (epoch, error, loss) in logbook]
             }
            for (hp, logbook) in logbooks]

    with open(params.statistics, 'w') as f:
        yaml.dump(stat, f)


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
                             hyper_parameter=learn_params.model_hyper_parameter,
                             **dataset.model_params)
        model.to(device)

        optimizer = optim.Adadelta(model.parameters())
        timer.stop_and_log()

    validation_ratio = 0.1
    validation_size = int(len(dataset) * validation_ratio)
    trainings_size = len(dataset) - validation_size

    train_set, val_set = torch.utils.data.random_split(dataset, [trainings_size, validation_size])

    # Loading data
    train_loader_params = {'batch_size': learn_params.batch_size,
                           'shuffle': True,
                           'num_workers': 0,
                           'collate_fn': dataset.collate_fn}

    validate_loader_params = {'batch_size': 8,
                              'shuffle': False,
                              'num_workers': 0,
                              'collate_fn': dataset.collate_fn}
    training_dataloader = data.DataLoader(train_set, **train_loader_params)
    validation_dataloader = data.DataLoader(val_set, **validate_loader_params)

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

    logbook = []

    timer = Timer(f'Training per sample:')
    model.train()
    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        model.zero_grad()
        for x, *s, y in training_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if len(s) > 0:
                s = s[0].to(device)
                x = model(x, s)
            else:
                x = model(x)
            # batch x tags
            loss = loss_function(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()
            epoch_loss += loss
        if epoch % exe_params.report_rate == 0:
            timer.pause()
            model.eval()
            error = validate(model, validation_dataloader)
            model.train()
            clearProgressBar()
            loss = learn_params.batch_size * epoch_loss
            logbook.append((epoch, error, loss))
            logging.info(
                f'#{epoch} Loss: {loss:.3f}  Error: {error.with_padding} (if rule: {error.when_rule}) exact: {error.exact} exact no padding: {error.exact_no_padding}')
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs)
    clearProgressBar()
    timer.stop_and_log_average(learn_params.num_epochs*len(dataset))

    signal.signal(signal.SIGINT, original_sigint_handler)
    error = validate(model, validation_dataloader)
    if logging.INFO >= logging.root.level:
        error.exact_no_padding.printHistogram()

    logbook.append((epoch, error, None))
    # dump_statistics(exe_params, logbook)
    save_snapshot()
    return logbook


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['training']


def load_model_hyperparameter(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['training']['model-parameter']


if __name__ == '__main__':

    parser = ArgumentParser('-c', '--config-file', loader=load_config,
                            prog='deep training')
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
    parser.add_argument('--statistics', default=None)

    # Learning parameter
    parser.add_argument('-n', '--num-epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-g', '--gradient-clipping', type=float, default=0.1)
    parser.add_argument('-m', '--model', choices=all_models,
                        default='TreeCnnSegmenter', dest='model_name')

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
    parser.add_argument('--data-size-limit', type=int, default=None,
                        help='Limits the size of the loaded bag file data. For testing purpose.')

    args = parser.parse_args()
    loglevel = 'INFO' if args.verbose else args.log.upper()
    if sys.stdin.isatty():
        log_format = '%(message)s'
    else:
        log_format = '%(asctime)s %(message)s'

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_hyper_parameters = load_model_hyperparameter(args.config)

    stats = []
    changed_parameters = grid_search.get_range_names(model_hyper_parameters, vars(args))
    for model_hyper_parameter, arg in grid_search.unroll_many(model_hyper_parameters, vars(args)):
        # If there might be conflicts in argument names use https://stackoverflow.com/a/18677482/6863221
        result = main(
            exe_params=ExecutionParameter(**vars(args)),
            learn_params=LearningParmeter(model_hyper_parameter=model_hyper_parameter,
                                          **arg),
            scenario_params=ScenarioParameter(**arg)
        )
        stats.append((grid_search.strip_keys(model_hyper_parameter, arg, names=changed_parameters), result))
    dump_statistics(ExecutionParameter(**vars(args)), stats)
