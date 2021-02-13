#!/usr/bin/env python3

import argparse
import math
import signal
import time
import sys
import logging
from pathlib import Path

import sherpa
import yaml

from common.parameter_search import LearningParmeter
from common.utils import make_namespace, setup_logging
from dataset import ScenarioParameter
from train import ExecutionParameter
from train import main as train
from t3_loop import main as t3_loop, create_parser as t3_parser
from solve import main as solve, create_parser as solve_parser

logger = logging.getLogger(__name__)


def power2(begin: int, end: int):
    begin = int(math.log2(begin))
    end = int(math.log2(end))+1
    return [2**e for e in range(begin, end)]


def create_training_parameters():
    return [
        sherpa.Choice('batch-size', power2(4, 32)),
        sherpa.Continuous('learning-rate', [0.001, 0.2], scale='log'),
        sherpa.Continuous('gradient-clipping', [0., 0.2], scale='linear'),
        sherpa.Continuous('value-loss-weight', [0.1, 0.9], scale='linear'),
        # Model parameters
        sherpa.Continuous('dropout', [0.1, 0.9], scale='linear'),
        sherpa.Choice('embedding_size', power2(8, 128)),
        sherpa.Choice('hidden_layers', [1, 2, 3, 4]),
        sherpa.Choice('use_props', [False, True]),
        sherpa.Choice('residual', [False, True]),
    ]


def touch(prefix: str):
    datestring = time.strftime('%b%d_%H-%M-%S')
    record_filename = Path(f'runs/hyperdrive/{datestring}-{prefix}/summary.yaml')

    logging.info(f'Touching to {record_filename} ...')
    record_filename.parent.mkdir(parents=True, exist_ok=True)
    record_filename.touch()
    return record_filename


def train_cmd(args, config):
    parameters = create_training_parameters()

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=args.max_num_trials)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=True)

    scenario_params = ScenarioParameter.from_config(config, use_solver_data=True)
    exe_params = ExecutionParameter(report_rate=config.training.report_rate, device='cpu', tensorboard=True,
                                    manual_seed=True, use_solved_problems=True, create_fresh_model=True)

    record_filename = touch('training')

    for i, trial in enumerate(study):
        error = {}

        def early_abort(epoch: int, metric: float):
            study.add_observation(
                trial=trial, objective=metric, iteration=epoch)
            error[epoch] = metric
            return not study.should_trial_stop(trial)

        learn_params = LearningParmeter.from_config(config)
        learn_params.learning_rate = trial.parameters['learning-rate']
        learn_params.batch_size = trial.parameters['batch-size']
        learn_params.gradient_clipping = trial.parameters['gradient-clipping']
        learn_params.value_loss_weight = trial.parameters['value-loss-weight']
        learn_params.model_hyper_parameter.update(trial.parameters)

        train(exe_params=exe_params, scenario_params=scenario_params,
              learn_params=learn_params, config=config,
              early_abort_hook=early_abort, no_sig_handler=True,
              tensorboard_dir=record_filename.parent / f'trial-{i}')

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump([{'hparams': trial.parameters, 'error': error}], f)


def t3loop_cmd(args, config):
    parameters = [
        sherpa.Choice('num-epochs', [30]),
        sherpa.Choice('problems.beam-size', list(range(3, 20))),
        sherpa.Choice('problems.num_epochs', list(range(10, 20))),
        sherpa.Choice('problems.max_track_loss', list(range(2, 5))),
        sherpa.Choice('problems.max_fit_results', [500, 1000, 1500]),
        sherpa.Choice('problems.iterations', [5, 7, 9]),
    ]

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=args.max_num_trials)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)

    record_filename = touch('t3loop')

    config_args, self_args = t3_parser().parse_args(['-c', str(args.config)])

    self_args.policy_last = True
    self_args.fresh_model = True

    for i, trial in enumerate(study):
        solving_rate = {}

        def early_abort(epoch: int, metric: float):
            study.add_observation(
                trial=trial, objective=metric, iteration=epoch)
            solving_rate[epoch] = metric
            return not study.should_trial_stop(trial)

        config_args.evaluation.num_epochs = trial.parameters['num-epochs']
        config_args.evaluation.problems.beam_size = trial.parameters['problems.beam-size']
        config_args.evaluation.problems.num_epochs = trial.parameters['problems.num_epochs']
        config_args.evaluation.problems.max_track_loss = trial.parameters['problems.max_track_loss']
        config_args.evaluation.problems.max_fit_results = trial.parameters['problems.max_fit_results']
        config_args.evaluation.problems.iterations = trial.parameters['problems.iterations']

        self_args.tensorboard_dir = record_filename.parent / f'trial-{i}'

        t3_loop(self_args, config_args, early_abort_hook=early_abort)

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump([{'hparams': trial.parameters, 'solving_ratio': solving_rate}], f)


def solve_cmd(args, config):
    parameters = [
        sherpa.Choice('num-epochs', [30]),
        sherpa.Choice('problems.beam-size', list(range(3, 20))),
        sherpa.Choice('problems.num_epochs', list(range(10, 20))),
        sherpa.Choice('problems.max_track_loss', list(range(2, 5))),
        sherpa.Choice('problems.max_fit_results', [500, 1000, 1500]),
        sherpa.Choice('problems.iterations', [5, 7, 9]),
    ]

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=args.max_num_trials)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)

    record_filename = touch('solve')

    config_args, self_args = solve_parser().parse_args(['-c', str(args.config)])

    self_args.policy_last = True
    self_args.fresh_model = True
    self_args.no_dumping = True

    for i, trial in enumerate(study):
        solving_rate = {}

        config_args.evaluation.num_epochs = trial.parameters['num-epochs']
        config_args.evaluation.problems.beam_size = trial.parameters['problems.beam-size']
        config_args.evaluation.problems.num_epochs = trial.parameters['problems.num_epochs']
        config_args.evaluation.problems.max_track_loss = trial.parameters['problems.max_track_loss']
        config_args.evaluation.problems.max_fit_results = trial.parameters['problems.max_fit_results']

        self_args.tensorboard_dir = record_filename.parent / f'solve-{i}'

        success_rate = solve(self_args, config_args)
        study.add_observation(trial=trial, objective=success_rate, iteration=0)

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump([{'hparams': trial.parameters, 'solving_ratio': solving_rate}], f)


def main(args):
    with args.config.open() as f:
        config = make_namespace(yaml.safe_load(f))
    args.func(args, config)


if __name__ == '__main__':
    setup_logging(verbose=False, log='warning')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default='real_world_problems/number_crunching/config.yaml')
    parser.add_argument('--max-num-trials', type=int, default=100)
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_cmd)
    parser_train = subparsers.add_parser('t3loop')
    parser_train.set_defaults(func=t3loop_cmd)
    parser_train = subparsers.add_parser('solve')
    parser_train.set_defaults(func=solve_cmd)

    main(parser.parse_args())
