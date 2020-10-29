#!/usr/bin/env python3

import logging
import signal
import sys
import yaml
from pathlib import Path

try:
    from azureml.core import Run
    import azureml
    azure_run = Run.get_context(allow_offline=False)
except ImportError:
    azure_run = None
except AttributeError:  # Running in offline mode
    azure_run = None
except azureml.exceptions.RunEnvironmentException:
    azure_run = None

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

logger = logging.getLogger(__name__)


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, update_model: str = None, load_model: str = None,
                 save_model: str = None, device: str = 'auto', tensorboard: bool = False, statistics: str = None,
                 manual_seed: bool = False, **kwargs):
        self.report_rate = report_rate
        self.load_model = load_model or update_model
        self.save_model = save_model or update_model
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.tensorboard = tensorboard
        self.statistics = statistics
        self.manual_seed = manual_seed

    @staticmethod
    def add_parsers(parser: ArgumentParser):
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
        parser.add_argument('--manual-seed', action='store_true', default=False)


def dump_statistics(params: ExecutionParameter, logbooks):
    '''Dumps statistics '''
    stat = [{'parameter': hp,
             'results': [{'error': error.as_dict(), 'epoch': epoch} for (epoch, error, loss) in logbook]
             }
            for (hp, logbook) in logbooks]

    filename = Path(params.statistics)
    filename.parent.mkdir(exist_ok=True)
    with filename.open('w') as f:
        yaml.dump(stat, f)

    if azure_run is not None:
        print(azure_run)
        for i, logbook in enumerate(logbooks):
            (_, error, loss) = logbook[-1][0]

            def sumup(name):
                ratio = getattr(error, name)
                return ratio.topk(5)

            row = {n: sumup(n) for n in ['exact', 'exact_no_padding']}
            azure_run.log_row(f'Parameter set {i+1}', **row)
        # Last error
        (_, last, _) = logbooks[-1][-1][0]
        top1 = last.exact_no_padding.topk(1)
        azure_run.log('top1', top1)
        top2 = last.exact_no_padding.topk(2)
        azure_run.log('top2', top2)
        top3 = last.exact_no_padding.topk(3)
        azure_run.log('top3', top3)


def main(exe_params: ExecutionParameter, learn_params: LearningParmeter, scenario_params: ScenarioParameter):
    if exe_params.manual_seed:
        torch.manual_seed(0)
    device = torch.device(exe_params.device)
    logger.info(f'Using device: {device}')

    # Creating dataset and model
    if exe_params.load_model:
        dataset, model, optimizer, _ = io.load(exe_params.exe_params.load_model, device)
    else:
        timer = Timer('Creating fresh workspace')
        dataset = create_scenario(params=scenario_params, device=device)
        model = create_model(learn_params.model_name,
                             hyper_parameter=learn_params.model_hyper_parameter,
                             **dataset.model_params)
        model.to(device)

        optimizer = optim.Adadelta(model.parameters(), lr=learn_params.learning_rate)
        timer.stop_and_log()

    def save_snapshot():
        io.save(exe_params.save_model, model, optimizer, scenario_params, learn_params, dataset)

    if len(dataset) == 0:
        logger.info('Loaded empty bagfile')
        save_snapshot()
        return []

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
    policy_loss_function = nn.NLLLoss(reduction='mean', weight=weight, ignore_index=-1)
    value_loss_function = nn.NLLLoss(reduction='mean')

    # Training
    original_sigint_handler = signal.getsignal(signal.SIGINT)

    def early_abort(*args):
        clearProgressBar()
        logger.warning('Early abort')
        save_snapshot()
        sys.exit(1)
    signal.signal(signal.SIGINT, early_abort)

    if exe_params.tensorboard:
        writer = SummaryWriter()
        x, s, _ = next(iter(validation_dataloader))
        writer.add_graph(model, (x, s))
        writer.close()

    logbook = []

    timer = Timer('Training per sample:')
    model.train()

    for epoch in range(learn_params.num_epochs):
        epoch_loss = 0
        model.zero_grad()
        for x, *s, y, p, v in training_dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if len(s) > 0:
                s = s[0].to(device)
                py, pv = model(x, s, p)
            else:
                py, pv = model(x)
            # batch x tags
            policy_loss = policy_loss_function(py, y)
            v = v.squeeze()
            # print(f'pv: {pv.shape}  v: {v.shape}')
            value_loss = value_loss_function(pv, v)
            loss = policy_loss + value_loss*learn_params.value_loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_value_(
                model.parameters(), learn_params.gradient_clipping)
            optimizer.step()
            epoch_loss += loss
        if epoch % exe_params.report_rate == 0:
            timer.pause()
            model.eval()
            error, value_error = validate(model, validation_dataloader)
            model.train()
            clearProgressBar()
            loss = learn_params.batch_size * epoch_loss
            logbook.append((epoch, error, loss))
            logger.info(
                f'#{epoch} Loss: {loss:.3f}  Error: {error.with_padding} (if rule: {error.when_rule}) exact: {error.exact} exact no padding: {error.exact_no_padding} value error: {value_error}')
            if azure_run is not None:
                error.exact.log(azure_run.log, 'exact')
                error.exact_no_padding.log(azure_run.log, 'exact (np)')
                error.with_padding.log(azure_run.log, 'class')
                error.when_rule.log(azure_run.log, 'class (np)')
                azure_run.log('loss', loss.item())
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs)
    clearProgressBar()
    duration_per_sample = timer.stop_and_log_average(learn_params.num_epochs*len(dataset))
    if azure_run is not None:
        azure_run.log('duration_per_sample', duration_per_sample)

    signal.signal(signal.SIGINT, original_sigint_handler)
    error, value_error = validate(model, validation_dataloader)
    if logging.INFO >= logging.root.level:
        error.exact_no_padding.printHistogram()

    logbook.append((learn_params.num_epochs, error, None))
    save_snapshot()
    return logbook


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['training']


def setup_loggin(verbose, log, **kwargs):
    loglevel = 'INFO' if verbose else log.upper()
    if sys.stdin.isatty():
        log_format = '%(message)s'
    else:
        log_format = '%(asctime)s %(message)s'

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )
    # Set the log level of all existing loggers
    all_loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for mod_logger in all_loggers:
        mod_logger._cache.clear()  # pylint: disable=protected-access
        mod_logger.setLevel(logging.getLevelName(loglevel))


if __name__ == '__main__':

    parser = ArgumentParser('-c', '--config-file', loader=load_config,
                            prog='deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Make a very fast run. (data_size_limit: 100, num_epochs: 1)')

    ExecutionParameter.add_parsers(parser)
    LearningParmeter.add_parsers(parser, all_models=all_models)
    ScenarioParameter.add_parsers(parser)
    # parser.add_argument('--optimizer', choices=[''])

    args, model_hyper_parameters = parser.parse_args()

    setup_loggin(**vars(args))

    if args.smoke:
        args.data_size_limit = 100
        args.num_epochs = 1

    stats = []
    changed_parameters = grid_search.get_range_names(model_hyper_parameters, vars(args))
    for model_hyper_parameter, arg in grid_search.unroll_many(model_hyper_parameters, vars(args)):
        logger.info(model_hyper_parameter)
        # If there might be conflicts in argument names use https://stackoverflow.com/a/18677482/6863221
        result = main(
            exe_params=ExecutionParameter(**vars(args)),
            learn_params=LearningParmeter(model_hyper_parameter=model_hyper_parameter,
                                          **arg),
            scenario_params=ScenarioParameter(**arg)
        )
        stats.append((grid_search.strip_keys(model_hyper_parameter, arg, names=changed_parameters), result))
    dump_statistics(ExecutionParameter(**vars(args)), stats)
