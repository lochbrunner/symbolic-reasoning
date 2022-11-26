#!/usr/bin/env python3

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sherpa
import yaml

from common.parameter_search import LearningParameter
from common.utils import make_namespace, setup_logging
from dataset import ScenarioParameter
from solve import create_parser as solve_parser
from solve import main as solve
from t3_loop import create_parser as t3_parser
from t3_loop import main as t3_loop
from train import ExecutionParameter
from train import main as train

logger = logging.getLogger(__name__)


def power2(begin: int, end: int):
    begin = int(math.log2(begin))
    end = int(math.log2(end)) + 1
    return [2**e for e in range(begin, end)]


def create_training_parameters():
    return [
        sherpa.Choice('batch-size', power2(4, 32)),
        sherpa.Continuous('learning-rate', [0.001, 0.2], scale='log'),
        sherpa.Continuous('gradient-clipping', [0.0, 0.2], scale='linear'),
        sherpa.Continuous('value-loss-weight', [0.1, 0.9], scale='linear'),
        # Model parameters
        sherpa.Continuous('dropout', [0.1, 0.9], scale='linear'),
        sherpa.Choice('embedding_size', power2(8, 128)),
        sherpa.Choice('hidden_layers', [1, 2, 3, 4]),
        sherpa.Choice('use_props', [False, True]),
        sherpa.Choice('residual', [False, True]),
    ]


def create_training_seed(config):
    return {
        'batch-size': config.training.batch_size,
        'learning-rate': config.training.learning_rate.initial,
        'gradient-clipping': config.training.gradient_clipping,
        'value-loss-weight': config.training.value_loss_weight,
        # Model parameters
        'dropout': config.training.model_parameter.dropout,
        'embedding_size': config.training.model_parameter.embedding_size,
        'hidden_layers': config.training.model_parameter.hidden_layers,
        'use_props': config.training.model_parameter.use_props,
        'residual': config.training.model_parameter.residual,
    }


def sanitize_numpy(param: dict):
    '''Because of any reason the fields can flip to numpy types'''

    def fix(value):
        if isinstance(value, np.int64):
            return int(value)
        if isinstance(value, np.float):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    return {k: fix(v) for k, v in param.items()}


def touch(prefix: str):
    datestring = time.strftime('%b%d_%H-%M-%S')
    record_filename = Path(f'runs/hyperdrive/{datestring}-{prefix}/summary.yaml')

    logging.info(f'Touching to {record_filename} ...')
    record_filename.parent.mkdir(parents=True, exist_ok=True)
    record_filename.touch()
    return record_filename


def create_algorithm(args, seed_configuration: dict):
    if args.algorithm == 'bayesian':
        return sherpa.algorithms.bayesian_optimization.GPyOpt(
            max_num_trials=args.max_num_trials
        )
    elif args.algorithm == 'local':
        return sherpa.algorithms.core.LocalSearch(seed_configuration=seed_configuration)
    else:
        raise RuntimeError(f'Not supported optimization algorithm: {args.algorithm}')


def train_cmd(args, config):
    parameters = create_training_parameters()

    algorithm = create_algorithm(args, create_training_seed(config))

    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=True
    )

    scenario_params = ScenarioParameter.from_config(config, use_solver_data=True)
    scenario_params.data_size_limit = 0.01
    exe_params = ExecutionParameter(
        training=config.training,
        device='cpu',
        tensorboard=True,
        manual_seed=True,
        use_solved_problems=True,
        create_fresh_model=True,
        dont_dump_model=True,
    )

    record_filename = touch('training')

    for i, trial in enumerate(study):
        error = {}

        def early_abort(epoch: int, metric: float):
            study.add_observation(trial=trial, objective=metric, iteration=epoch)
            error[epoch] = metric
            return not study.should_trial_stop(trial)

        learn_params = LearningParameter.from_config(config)
        params = sanitize_numpy(trial.parameters)

        learn_params.learning_rate = params['learning-rate']
        learn_params.batch_size = params['batch-size']
        learn_params.gradient_clipping = params['gradient-clipping']
        learn_params.value_loss_weight = params['value-loss-weight']
        learn_params.model_hyper_parameter.update(params)

        train(
            exe_params=exe_params,
            scenario_params=scenario_params,
            learn_params=learn_params,
            config=config,
            early_abort_hook=early_abort,
            no_sig_handler=True,
            tensorboard_dir=record_filename.parent / f'trial-{i}',
        )

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump([{'hparams': params, 'error': error}], f)


def t3loop_cmd(args, config):
    parameters = [
        sherpa.Choice('num-epochs', [30]),
        sherpa.Choice('problems.beam-size', list(range(3, 20))),
        sherpa.Choice('problems.num_epochs', list(range(10, 20))),
        sherpa.Choice('problems.max_track_loss', list(range(2, 5))),
        sherpa.Choice('problems.max_fit_results', [500, 1000, 1500]),
        sherpa.Choice('problems.iterations', [5, 7, 9]),
    ]

    seed_configuration = {
        'num-epochs': 30,
        'problems.beam-size': 15,
        'problems.num_epochs': 15,
        'problems.max_track_loss': 3,
        'problems.max_fit_results': 1000,
        'problems.iterations': 7,
    }

    algorithm = create_algorithm(args, seed_configuration)

    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=False
    )

    record_filename = touch('t3loop')

    config_args, self_args = t3_parser().parse_args(['-c', str(args.config)])

    self_args.policy_last = True
    self_args.fresh_model = True

    for i, trial in enumerate(study):
        solving_rate = {}

        def early_abort(epoch: int, metric: float):
            study.add_observation(trial=trial, objective=metric, iteration=epoch)
            solving_rate[epoch] = metric
            return not study.should_trial_stop(trial)

        config_args.evaluation.num_epochs = trial.parameters['num-epochs']
        config_args.evaluation.problems.beam_size = trial.parameters[
            'problems.beam-size'
        ]
        config_args.evaluation.problems.num_epochs = trial.parameters[
            'problems.num_epochs'
        ]
        config_args.evaluation.problems.max_track_loss = trial.parameters[
            'problems.max_track_loss'
        ]
        config_args.evaluation.problems.max_fit_results = trial.parameters[
            'problems.max_fit_results'
        ]
        config_args.evaluation.problems.iterations = trial.parameters[
            'problems.iterations'
        ]

        self_args.tensorboard_dir = record_filename.parent / f'trial-{i}'

        t3_loop(self_args, config_args, early_abort_hook=early_abort)

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump(
                [{'hparams': trial.parameters, 'solving_ratio': solving_rate}], f
            )


def solve_cmd(args, config):
    parameters = [
        # sherpa.Choice('num-epochs', [30]),
        # sherpa.Choice('problems.beam-size', list(range(3, 20))),
        # sherpa.Choice('problems.num_epochs', list(range(10, 20))),
        # sherpa.Choice('problems.max_track_loss', list(range(2, 5))),
        sherpa.Choice('problems.max_fit_results', list(range(1000, 7000, 1000))),
        # sherpa.Choice('problems.iterations', [5, 7, 9]),
    ]

    seed_configuration = {'problems.max_fit_results': 1000}

    algorithm = create_algorithm(args, seed_configuration)

    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=False
    )

    record_filename = touch('solve')

    config_args, self_args = solve_parser().parse_args(['-c', str(args.config)])

    self_args.policy_last = True
    self_args.fresh_model = True
    self_args.no_dumping = True
    self_args.solve_training = True

    for i, trial in enumerate(study):
        solving_rate = {}

        # config_args.evaluation.num_epochs = trial.parameters['num-epochs']
        # config_args.evaluation.problems.beam_size = trial.parameters['problems.beam-size']
        # config_args.evaluation.problems.num_epochs = trial.parameters['problems.num_epochs']
        # config_args.evaluation.problems.max_fit_results = trial.parameters['problems.max_track_loss']
        config_args.evaluation.problems.max_fit_results = trial.parameters[
            'problems.max_fit_results'
        ]

        self_args.tensorboard_dir = record_filename.parent / f'solve-{i}'

        success_rate = solve(self_args, config_args)
        logger.info(f'{success_rate} with {trial.parameters}')
        study.add_observation(trial=trial, objective=success_rate, iteration=0)

        study.finalize(trial=trial)

        with record_filename.open('a') as f:
            yaml.safe_dump(
                [{'hparams': trial.parameters, 'solving_ratio': solving_rate}], f
            )


def main(args):
    with args.config.open() as f:
        config = make_namespace(yaml.safe_load(f))
    args.func(args, config)


if __name__ == '__main__':
    setup_logging(verbose=False, log='info')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default='real_world_problems/number_crunching/config.yaml',
    )
    parser.add_argument('--max-num-trials', type=int, default=100)
    parser.add_argument('--algorithm', default='local', choices=['local', 'bayesian'])
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train_cmd)
    parser_train = subparsers.add_parser('t3loop')
    parser_train.set_defaults(func=t3loop_cmd)
    parser_train = subparsers.add_parser('solve')
    parser_train.set_defaults(func=solve_cmd)

    main(parser.parse_args())
