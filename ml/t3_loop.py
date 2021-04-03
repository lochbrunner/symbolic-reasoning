#!/usr/bin/env python3

import logging
from pathlib import Path

import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import grid_search, io
from common.config_and_arg_parser import ArgumentParser
from common.parameter_search import LearningParmeter
from common.reports import report_tops
from common.timer import Timer
from dataset.bag import BagDataset
from pycore import ProblemStatistics, Scenario, SolverStatistics, Trace
from solver.inferencer import Inferencer
from solver.solve_problems import solve_problems
from solver.trace import TrainingsDataDumper
from solver.metrics import SolutionSummarieser
from training.validation import Mean
from training import train

logger = logging.getLogger(__name__)


def main(options, config, early_abort_hook=None):
    with Timer('Loading scenario'):
        scenario = Scenario.load(config.files.scenario)

    try:
        if options.tensorboard_dir:
            options.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(options.tensorboard_dir))
        else:
            writer = None

        # model, idents, rules = io.load_model(model, depth=9)
        rule_mapping = {}
        used_rules = set()
        max_width = max(len(s.name) for s in scenario.rules)+1
        for i, rule in enumerate(scenario.rules, 1):
            rule_mapping[i] = rule
            used_rules.add(str(rule))
            logger.debug(f'Using rule {i:2}# {rule.name.ljust(max_width)} {rule.verbose}')

        logger.info(f'Using {len(used_rules)} rules.')

        for scenario_rule in scenario.rules:
            if str(scenario_rule) not in used_rules:
                logger.warning(f'The rule "{scenario_rule}" was not in the model created by the training.')

        inferencer = Inferencer(config=config, scenario=scenario, fresh_model=options.fresh_model)
        solver_logger = logging.Logger('solver')
        solver_logger.setLevel(logging.WARNING)

        learn_params = LearningParmeter.from_config(config)
        data_loader_config = {'batch_size': learn_params.batch_size,
                              'shuffle': True,
                              'drop_last': True,
                              'num_workers': 0,
                              'collate_fn': BagDataset.collate_fn}

        optimizer = optim.Adadelta(inferencer.model.parameters(), lr=learn_params.learning_rate)

        trainings_data_dumper = TrainingsDataDumper(config, scenario)
        problems_statistics = {problem.name: ProblemStatistics(problem.name, problem.conclusion.latex)
                               for problem in scenario.problems.training}

        for iteration in range(config.evaluation.problems.iterations):
            # Try
            use_network = not options.fresh_model or iteration > 0
            solution_summarieser = SolutionSummarieser()

            success_rate = Mean()
            needed_fits = Mean()
            for statistics, solution in solve_problems(
                    options, config, scenario.problems.training, inferencer, rule_mapping, logger=solver_logger, use_network=use_network):
                if solution is not None:
                    solution_summarieser += solution

            # Trace
                success_rate += statistics.success
                if statistics.success:
                    trainings_data_dumper += statistics
                    needed_fits += statistics.fit_results

                problems_statistics[statistics.name] += statistics.as_builtin

            tops = solution_summarieser.summary()
            report_tops(tops['policy'], epoch=iteration, writer=writer, label='policy')
            report_tops(tops['value'], epoch=iteration, writer=writer, label='value')

            if writer:
                writer.add_scalar('solved/relative', success_rate.summary, iteration)
                writer.add_scalar('needed-fits', needed_fits.summary, iteration)
            if success_rate.correct == 0.0:
                logger.warning('Could not solve any of the training problems.')
                return
            logger.info(f'Solved: {success_rate.verbose} with {needed_fits.statistic} fits in iteration {iteration}')
            dataset = trainings_data_dumper.get_dataset()
            logger.info(f'Training with {len(dataset)} samples')
            if len(dataset) < 2:
                raise RuntimeError(f'Too less samples. Just {len(dataset)} available')
            training_dataloader = data.DataLoader(dataset, **data_loader_config)

            # Train
            if options.just_dump_trainings_data:
                break
            if options.smoke:
                learn_params.num_epochs = 1

            train(learn_params=learn_params, model=inferencer.model, optimizer=optimizer,
                  training_dataloader=training_dataloader, policy_weight=dataset.label_weight, value_weight=dataset.value_weight)

            if early_abort_hook and early_abort_hook(iteration, float(success_rate)):
                break

            learn_params.use_finetuning()

    finally:
        if writer:
            writer.close()

    intro = SolverStatistics()
    for stat in problems_statistics.values():
        intro += stat
    logger.info(f'Dumping traces to "{config.files.t3_loop_traces}" ...')
    with tqdm(total=100, desc='Create index', leave=False) as progress_bar:
        intro.create_index(lambda progress: progress_bar.update(min(100*progress, 100)))
    intro.dump(config.files.t3_loop_traces)

    trainings_data_dumper.dump(rule_mapping)


def create_parser():
    parser = ArgumentParser(domain='evaluation', prog='solver', exclude="scenario-*")
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--results-filename')
    parser.add_argument('--policy-last', action='store_true', default=False)
    parser.add_argument('--tensorboard-dir', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Run only a the first samples to test the functionality.')
    parser.add_argument('--just-dump-trainings-data', action='store_true', default=False,
                        help='Just try, trace and dump trainings data.')

    # Model
    parser.add_argument('--fresh-model', action='store_true', help='Creates a fresh model')
    return parser


if __name__ == '__main__':

    config_args, self_args = create_parser().parse_args()
    loglevel = 'INFO' if self_args.verbose else self_args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )

    main(self_args, config_args)
