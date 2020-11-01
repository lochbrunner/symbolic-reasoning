import logging
from typing import Dict

from pycore import Symbol, Scenario

from common.timer import Timer
from common.terminal_utils import printProgressBar, clearProgressBar
from solver.beam_search import beam_search, beam_search_policy_last
from solver.inferencer import Inferencer
from solver.trace import TrainingsDataDumper, dump_new_rules


module_logger = logging.getLogger(__name__)


def solve_problems(options, config, scenario: Scenario, inferencer: Inferencer,
                   rule_mapping: Dict, logger: logging.Logger = module_logger):
    problem_statistics = []
    problem_solutions = []

    show_progress = not logger.isEnabledFor(logging.INFO)

    context = scenario.declarations

    def variable_generator():
        return Symbol.parse(context, 'u')

    eval_config = config.evaluation

    for i, problem_name in enumerate(scenario.problems):
        if show_progress:
            printProgressBar(i, len(scenario.problems))
        problem = scenario.problems[problem_name]
        # Get first problem
        logger.info(f'problem: {problem}')
        logger.info('-'*30)
        source = problem.condition
        target = problem.conclusion

        with Timer(f'Solving problem "{problem_name}"', logger=logger):
            search_strategy = beam_search_policy_last if options.policy_last else beam_search
            solution, statistics = search_strategy(inferencer, rule_mapping, problem.condition,
                                                   [problem.conclusion], variable_generator,
                                                   beam_size=eval_config.problems.beam_size,
                                                   num_epochs=eval_config.problems.num_epochs,
                                                   black_list_terms=eval_config.black_list_terms,
                                                   black_list_rules=eval_config.black_list_rules,
                                                   max_size=eval_config.max_size)
        if solution is not None:
            problem_solutions.append(solution)

        else:
            logger.info(f'No solution found for {source} => {target}')

        statistics.name = problem_name
        logger.info(statistics)
        problem_statistics.append(statistics)

    if show_progress:
        clearProgressBar()

    dump_new_rules(solutions=problem_solutions, new_rules_filename=config.evaluation.new_rules_filename)

    return problem_solutions, problem_statistics
