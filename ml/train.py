#!/usr/bin/env python3

import logging
import signal
import sys
import subprocess as sp
import yaml
from pathlib import Path

try:
    from azureml.core import Run
    import azureml
    azure_run = Run.get_context(allow_offline=False)
    from azureml.tensorboard import Tensorboard
except ImportError:
    azure_run = None
except AttributeError:  # Running in offline mode
    azure_run = None
except azureml.exceptions.RunEnvironmentException:
    azure_run = None

import torch
import torch.optim as optim
from torch.utils import data
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    pass

from dataset import create_scenario, ScenarioParameter
from models import create_model

from common import grid_search
from common import io
from common.config_and_arg_parser import ArgumentParser
from common.parameter_search import LearningParmeter
from common.timer import Timer
from common.utils import setup_logging, get_rule_mapping_by_config, split_dataset
from training.validation import validate
from training import train

logger = logging.getLogger(__name__)


class ExecutionParameter:
    def __init__(self, device: str, tensorboard: bool, training: object,
                 manual_seed: bool, use_solved_problems: bool, create_fresh_model: bool,
                 dont_dump_model: bool = False, just_dump_model: bool = False, **kwargs):
        self.report_rate = training.report_rate
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.tensorboard = tensorboard
        self.manual_seed = manual_seed
        self.use_solved_problems = use_solved_problems
        self.create_fresh_model = create_fresh_model
        self.dont_dump_model = dont_dump_model
        self.just_dump_model = just_dump_model

    @staticmethod
    def add_parsers(parser: ArgumentParser):
        parser.add_argument('-d', '--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
        parser.add_argument('--tensorboard', action='store_true', default=False)
        parser.add_argument('--manual-seed', action='store_true', default=False)
        parser.add_argument('--create-fresh-model', action='store_true', default=False)
        parser.add_argument('--use-solved-problems', action='store_true', default=False)
        parser.add_argument('--dont-dump-model', action='store_true', default=False)
        parser.add_argument('--just-dump-model', action='store_true', help='Overides dont dump model', default=False)


def dump_statistics(config, logbooks):
    '''Dumps statistics '''
    stat = [{'parameter': hp,
             'results': [{'error': error.as_dict(), 'epoch': epoch} for (epoch, error, loss) in logbook]
             }
            for (hp, logbook) in logbooks]

    filename = Path(config.files.training_statistics)
    filename.parent.mkdir(exist_ok=True)
    with filename.open('w') as f:
        yaml.dump(stat, f)


def main(exe_params: ExecutionParameter, learn_params: LearningParmeter,
         scenario_params: ScenarioParameter, config,
         early_abort_hook=None, no_sig_handler=False, tensorboard_dir=None):
    if exe_params.manual_seed:
        torch.manual_seed(0)
    device = torch.device(exe_params.device)
    logger.info(f'Using device: {device}')

    # Creating dataset and model
    if exe_params.create_fresh_model:
        timer = Timer('Creating fresh workspace')
        if exe_params.use_solved_problems:
            scenario_params.filename = config.files.solver_trainings_data
        dataset = create_scenario(params=scenario_params, device=device)
        model = create_model(learn_params.model_name,
                             hyper_parameter=learn_params.model_hyper_parameter,
                             **dataset.model_params)
        model.to(device)
        if learn_params.optimizer == 'adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=learn_params.learning_rate,
                                       weight_decay=learn_params.weight_decay)
        elif learn_params.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learn_params.learning_rate,
                                   weight_decay=learn_params.weight_decay)
        else:
            raise RuntimeError(f'Unknown optimizer "{learn_params.optimizer}"')
        if learn_params.learning_rate_step_size is not None:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=learn_params.learning_rate_step_size,
                                                  gamma=learn_params.learning_rate_gamma)
        else:
            scheduler = None
        timer.stop_and_log()
    else:
        dataset, model, optimizer, _ = io.load(config.files.model, device)
        scheduler = None

    def save_snapshot(metrics=None):
        if not exe_params.dont_dump_model or exe_params.just_dump_model:
            io.save(config.files.model, model, optimizer, scenario_params, learn_params, dataset, metrics=metrics)
        else:
            logger.info(f'Skipping model dump to "{config.files.model}" as requested.')

    if exe_params.just_dump_model:
        save_snapshot()
        logger.warning('I just dump the model and exit. Bye bye!')
        return

    if len(dataset) == 0:
        logger.info('Loaded empty bagfile. Skip training.')
        save_snapshot()
        return []

    rule_mapping = get_rule_mapping_by_config(config)
    rule_mapping = {k: rule.name for k, rule in rule_mapping.items()}
    rule_mapping[0] = 'padding'

    # validation_ratio = 0.1
    # validation_size = int(len(dataset) * validation_ratio)
    # trainings_size = len(dataset) - validation_size

    # generator = torch.Generator()
    # generator.manual_seed(0)
    # train_set, val_set = torch.utils.data.random_split(dataset, [trainings_size, validation_size], generator=generator)

    train_set, val_set = split_dataset(dataset)

    # Loading data
    common_loader_params = {
        'num_workers': 0,
        'drop_last': True,
        'pin_memory': exe_params.device == 'cuda',
        'collate_fn': dataset.get_collate_fn()
    }
    train_loader_params = {'batch_size': int(learn_params.batch_size),
                           'shuffle': True,
                           **common_loader_params}

    validate_loader_params = {'batch_size': 8,
                              'shuffle': False,
                              **common_loader_params}
    training_dataloader = data.DataLoader(train_set, **train_loader_params)
    validation_dataloader = data.DataLoader(val_set, **validate_loader_params)

    policy_weight = torch.as_tensor(dataset.label_weight, device=device, dtype=torch.float)
    value_weight = torch.as_tensor(dataset.value_weight, device=device, dtype=torch.float)

    # Training
    if not no_sig_handler:
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def early_abort(*args):
            logger.warning('Abort by user')
            save_snapshot()
            sys.exit(1)
        signal.signal(signal.SIGINT, early_abort)

    try:
        tb = None
        if exe_params.tensorboard:
            if azure_run is not None:
                log_dir = Path('output/tensorboard')
                log_dir.mkdir(parents=True, exist_ok=True)
                try:
                    tb = Tensorboard([azure_run], local_root=str(log_dir), port=6006)
                    tb.start()
                except Exception as e:
                    logger.warning(f'Could not start tensorboard: {e}')
                writer = SummaryWriter(log_dir=str(log_dir))
            else:
                writer = SummaryWriter(tensorboard_dir and str(tensorboard_dir))
            sp.Popen(['tensorboard', '--logdir', writer.log_dir])
            x, s, _, p, _, _, _ = next(iter(validation_dataloader))
            device = model.device
            x = x.to(device)
            s = s.to(device)
            p = p.to(device)
            writer.add_graph(model, (x, s, p))

        else:
            writer = None

        logbook = []

        timer = Timer('Training per sample:')
        model.train()

        def report(epoch, epoch_loss, **kwargs):
            if epoch % exe_params.report_rate != 0:
                return True
            model.eval()
            validation = validate(model, validation_dataloader, no_negative=False, **kwargs)
            validation_training = validate(model, training_dataloader, no_negative=False, **kwargs)
            error = validation.error
            model.train()
            loss = epoch_loss and learn_params.batch_size * epoch_loss
            logbook.append((epoch, error, loss))
            if not writer and loss is not None:
                logger.info(
                    f'#{epoch} Loss: {loss:.3f}  Error: {error.with_padding} (if rule: {error.when_rule}) exact: {error.exact} exact no padding: {error.exact_no_padding} value error (all): {error.value_all}')
            if azure_run is not None:
                error.exact.log(azure_run.log, 'exact')
                error.exact_no_padding.log(azure_run.log, 'exact (np)')
                error.in_possibilities_positive.log(azure_run.log, 'just possibilities')
                error.in_possibilities_negative.log(azure_run.log, 'just possibilities (negative)')
                error.with_padding.log(azure_run.log, 'class')
                error.when_rule.log(azure_run.log, 'class (np)')
                if loss is not None:
                    azure_run.log('loss/training', loss)
                azure_run.log('loss/validation/policy', validation.policy_loss)
                azure_run.log('loss/validation/value', validation.value_loss)
                azure_run.log('value/all', float(error.value_all))
                azure_run.log('value/positive', float(error.value_positive))
                azure_run.log('value/negative', float(error.value_negative))
                azure_run.log('optimizer/learning rate', optimizer.param_groups[0]['lr'])

            if writer:
                error.exact.log_bundled(writer, 'policy/exact with padding', epoch)
                error.exact_no_padding.log_bundled(writer, 'policy/exact', epoch)
                error.in_possibilities_positive.log_bundled(writer, 'policy/just possibilities', epoch)
                error.in_possibilities_negative.log_bundled(writer, 'policy/just possibilities (negative)', epoch)

                train_error = validation_training.error
                train_error.exact.log_bundled(writer, 'train-policy/exact with padding', epoch)
                train_error.exact_no_padding.log_bundled(writer, 'train-policy/exact', epoch)
                train_error.in_possibilities_positive.log_bundled(writer, 'train-policy/just possibilities', epoch)
                train_error.in_possibilities_negative.log_bundled(
                    writer, 'train-policy/just possibilities (negative)', epoch)
                # error.with_padding.log_bundled(writer, 'policy/class', epoch)
                # error.when_rule.log_bundled(writer, 'policy/class (no padding)', epoch)
                if loss is not None:
                    writer.add_scalar('loss/training', loss, epoch)
                writer.add_scalar('loss/validation/policy', validation.policy_loss, epoch)
                writer.add_scalar('loss/validation/value', validation.value_loss, epoch)
                writer.add_scalar('value/all', float(error.value_all), epoch)
                writer.add_scalar('value/positive', float(error.value_positive), epoch)
                writer.add_scalar('value/negative', float(error.value_negative), epoch)
                writer.add_scalar('optimizer/learning rate', optimizer.param_groups[0]['lr'], epoch)

                writer.add_scalars('distribution/rule',
                                   {rule_mapping[i]: c for i, c in enumerate(validation.predicted_rule_distribution)}, epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)

            if early_abort_hook is not None:
                # Primary metric
                primary_metric = 1 - error.exact_no_padding.topk(4)
                return early_abort_hook(epoch, primary_metric)
            return True

        train(learn_params=learn_params, model=model, optimizer=optimizer, scheduler=scheduler,
              training_dataloader=training_dataloader, policy_weight=policy_weight, value_weight=value_weight, report_hook=report, azure_run=azure_run)

        if not no_sig_handler:
            signal.signal(signal.SIGINT, original_sigint_handler)
        error = validate(model, validation_dataloader, no_negative=False).error

        logbook.append((learn_params.num_epochs, error, None))
        save_snapshot(error)
        if writer:
            writer.add_hparams(hparam_dict={
                'batch-size': learn_params.batch_size,
                'learning-rate': learn_params.learning_rate,
                'gradient-clipping': learn_params.gradient_clipping,
                'value-lossweight': learn_params.value_loss_weight,
                'optimizer': learn_params.optimizer,
                ** learn_params.model_hyper_parameter
            },
                metric_dict={'kpi/value-all': float(error.value_all),
                             'kpi/exact-no-padding (1)': error.exact_no_padding.topk(1),
                             'kpi/exact-no-padding (2)': error.exact_no_padding.topk(2),
                             'kpi/exact-no-padding (3)': error.exact_no_padding.topk(3),
                             'kpi/exact-no-padding (5)': error.exact_no_padding.topk(5),
                             'kpi/exact-no-padding (9)': error.exact_no_padding.topk(9),
                             })
        return logbook

    finally:
        if writer:
            writer.close()
            if tb:
                tb.stop()


if __name__ == '__main__':

    parser = ArgumentParser('-c', '--config-file', exclude="scenario-*",
                            prog='deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--smoke', action='store_true', default=False,
                        help='Make a very fast run. (data_size_limit: 100, num_epochs: 1)')

    ExecutionParameter.add_parsers(parser)

    config, options = parser.parse_args()

    setup_logging(**vars(options))

    if options.smoke:
        config.training.data_size_limit = 100
        config.training.num_epochs = 1
        options.dont_dump_model = True

    stats = []
    model_hyper_parameters = config.training.model_parameter
    changed_parameters = grid_search.get_range_names(model_hyper_parameters, vars(config))
    for model_hyper_parameter, arg in grid_search.unroll_many(model_hyper_parameters, vars(config)):
        logger.info(model_hyper_parameter)
        # If there might be conflicts in argument names use https://stackoverflow.com/a/18677482/6863221
        result = main(
            exe_params=ExecutionParameter(**vars(config), **vars(options)),
            learn_params=LearningParmeter.from_config_and_hyper(
                config=config, model_hyper_parameter=model_hyper_parameter),
            scenario_params=ScenarioParameter.from_config(config),
            config=config
        )
        if options.just_dump_model:
            break
        stats.append((grid_search.strip_keys(model_hyper_parameter, arg, names=changed_parameters), result))
    dump_statistics(config=config, logbooks=stats)
