#!/usr/bin/env python3

import logging
from tqdm import tqdm

from pycore import ProblemStatistics, Scenario, SolverStatistics

from common.config_and_arg_parser import ArgumentParser
from common.timer import Timer
from training.validation import Mean
from common.utils import get_rule_mapping, setup_logging
from solver.inferencer import TorchInferencer, SophisticatedInferencer
from solver.solve_problems import solve_problems
from solver.trace import TrainingsDataDumper

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
    use_network = (
        not options.fresh_model and not config.evaluation.problems.shuffle_fits
    )  # TODO: use enum?

    rule_mapping = get_rule_mapping(scenario)

    if options.inferencer == 'sophisticated':
        inferencer = SophisticatedInferencer.from_scenario(scenario=scenario)
    else:
        inferencer = TorchInferencer(
            config=config, scenario=scenario, fresh_model=options.fresh_model
        )

    if scenario.problems is not None:
        problems = (
            scenario.problems.training
            if options.solve_training
            else scenario.problems.validation
        )
    else:
        raise AssertionError('No problems found in Scenario!')
    problems_statistics = {
        problem.name: ProblemStatistics(problem.name, problem.conclusion.latex)
        for problem in problems
    }
    trainings_data_dumper = TrainingsDataDumper(config, scenario)

    success_rate = Mean()
    needed_fits = Mean()
    for statistics, _ in solve_problems(
        options,
        config,
        problems,
        inferencer,
        rule_mapping,
        use_network=use_network,
        exploration_ratio=config.evaluation.exploration_ratio,
    ):

        # Trace
        success_rate += statistics.success
        if statistics.success:
            needed_fits += statistics.fit_results
            trainings_data_dumper += statistics

        problems_statistics[statistics.name] += statistics.as_builtin
        # Trace

    if success_rate.correct == 0.0:
        logger.warning('Could not solve any of the training problems.')
        return None
    logger.info(f'Solved: {success_rate.verbose} with {needed_fits.statistic} fits')

    if not options.no_dumping:
        intro = SolverStatistics()
        for stat in problems_statistics.values():
            intro += stat
        logger.info(f'Dumping traces to "{config.files.solver_traces}" ...')
        with tqdm(total=100, desc='Create index', leave=False) as progress_bar:
            intro.create_index(lambda progress: progress_bar.update(100 * progress))
        intro.dump(config.files.solver_traces)

    if options.dump_trainings_data:
        trainings_data_dumper.dump(rule_mapping)

    return success_rate.summary


def create_parser():
    parser = ArgumentParser(domain='evaluation', prog='solver', exclude='scenario-*')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument(
        '-o',
        '--once',
        action='store_true',
        default=False,
        help='Hide repeating messages',
    )
    parser.add_argument(
        '--solve-training',
        help='Tries to solve the trainings data',
        action='store_true',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run only a the first samples to test the functionality.',
    )
    parser.add_argument(
        '--no-dumping',
        action='store_true',
        default=False,
        help='Prevent from dumping traces.',
    )
    parser.add_argument(
        '--dump-trainings-data',
        action='store_true',
        default=False,
        help='Dumps trainings data.',
    )
    parser.add_argument(
        '--inferencer',
        default='sophisticated',
        choices=['sophisticated', 'torch'],
        help='The inferencer to use.',
    )
    parser.add_argument(
        '--search-strategy',
        default='beam-search',
        choices=['beam-search', 'beam-search-policy-last', 'bidirectional-beam-search'],
        help='The search strategy to use.',
    )

    # Model
    parser.add_argument(
        '--fresh-model', action='store_true', help='Creates a fresh model'
    )

    return parser


if __name__ == '__main__':
    config_args, self_args = create_parser().parse_args()
    setup_logging(verbose=self_args.verbose, log=self_args.log, once=self_args.once)
    metric = main(self_args, config_args)

    if azure_run is not None:
        azure_run.log('success/rate', metric)
