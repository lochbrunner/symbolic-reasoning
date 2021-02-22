#!/usr/bin/env python3

import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from common.config_and_arg_parser import ArgumentParser
from common.timer import Timer
from common.validation import Mean
from common.utils import get_rule_mapping
from pycore import ProblemStatistics, Scenario, SolverStatistics, Trace
from solver.inferencer import Inferencer
from solver.solve_problems import solve_problems

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

logger = logging.getLogger(__name__)


def main(options, config):
    # Try
    with Timer('Loading scenario'):
        scenario = Scenario.load(config.files.scenario)
    use_network = not options.fresh_model

    rule_mapping = get_rule_mapping(scenario)

    if use_network:
        inferencer = Inferencer(config=config, scenario=scenario, fresh_model=options.fresh_model)
    else:
        inferencer = None

    problems = scenario.problems.training if options.solve_training else scenario.problems.validation
    problems_statistics = {problem.name: ProblemStatistics(problem.name, problem.conclusion.latex)
                           for problem in problems}

    success_rate = Mean()
    needed_fits = Mean()
    for statistics, _ in solve_problems(
            options, config, problems, inferencer, rule_mapping, use_network=use_network):

        # Trace
        success_rate += statistics.success
        if statistics.success:
            needed_fits += statistics.fit_results

        problems_statistics[statistics.name] += statistics.as_builtin
        # Trace

    if success_rate.correct == 0.0:
        logger.warning('Could not solve any of the training problems.')
        return
    logger.info(f'Solved: {success_rate.verbose} with {needed_fits.statistic} fits')

    if not options.no_dumping:
        intro = SolverStatistics()
        for stat in problems_statistics.values():
            intro += stat
        logger.info(f'Dumping traces to "{config.files.solver_traces}" ...')
        with tqdm(total=100, desc='Create index', leave=False) as progress_bar:
            intro.create_index(lambda progress: progress_bar.update(100*progress))
        intro.dump(config.files.solver_traces)

    return success_rate.summary


def create_parser():
    parser = ArgumentParser(domain='evaluation', prog='solver', exclude='scenario-*')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--policy-last', action='store_true', default=False)
    parser.add_argument('--smoke', action='store_true', help='Run only a the first samples to test the functionality.')
    parser.add_argument('--no-dumping', action='store_true', default=False, help='Prevent from dumping traces.')

    # Model
    parser.add_argument('--fresh-model', action='store_true', help='Creates a fresh model')

    return parser


def setup_logging(config_args, self_args):
    loglevel = 'DEBUG' if self_args.verbose else self_args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for module_logger in loggers:
        module_logger.setLevel(logging.INFO)

    return config_args, self_args


if __name__ == '__main__':
    config_args, self_args = create_parser().parse_args()
    setup_logging(config_args, self_args)
    metric = main(self_args, config_args)

    if azure_run is not None:
        azure_run.log('success/rate', metric)
