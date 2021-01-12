#!/usr/bin/env python3

import sherpa
import math
import argparse
import yaml
from pathlib import Path

from train import ExecutionParameter, main as train
from common.parameter_search import LearningParmeter
from common.utils import make_namespace
from dataset import ScenarioParameter


def power2(begin: int, end: int):
    begin = int(math.log2(begin))
    end = int(math.log2(end))+1
    return [2**e for e in range(begin, end)]


def main(args):
    with args.config.open() as f:
        config = make_namespace(yaml.safe_load(f))
    parameters = [
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

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=args.max_num_trials)

    study = sherpa.Study(parameters=parameters,
                         algorithm=algorithm,
                         lower_is_better=False)

    scenario_params = ScenarioParameter.from_config(config, use_solver_data=True)
    exe_params = ExecutionParameter(report_rate=config.training.report_rate, device='cpu', tensorboard=True,
                                    manual_seed=True, use_solved_problems=True, create_fresh_model=True)
    for trial in study:

        def early_abort(epoch: int, metric: float):
            study.add_observation(
                trial=trial, objective=metric, iteration=epoch)
            return study.should_trial_stop(trial)

        learn_params = LearningParmeter.from_config(config)
        learn_params.learning_rate = trial.parameters['learning-rate']
        learn_params.batch_size = trial.parameters['batch-size']
        learn_params.gradient_clipping = trial.parameters['gradient-clipping']
        learn_params.value_loss_weight = trial.parameters['value-loss-weight']
        learn_params.model_hyper_parameter.update(trial.parameters)

        train(exe_params=exe_params, scenario_params=scenario_params,
              learn_params=learn_params, config=config, early_abort_hook=early_abort)

        study.finalize(trial=trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default='real_world_problems/number_crunching/config.yaml')
    parser.add_argument('--max-num-trials', type=int, default=100)
    main(parser.parse_args())
