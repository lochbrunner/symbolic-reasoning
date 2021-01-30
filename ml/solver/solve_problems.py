from typing import Dict
import itertools
import logging
from tqdm import tqdm
import sys

from pycore import Symbol, Context, Rule

from common.timer import Timer
from solver.beam_search import beam_search, beam_search_policy_last
from solver.inferencer import Inferencer


module_logger = logging.getLogger(__name__)


def solve_problems(options, config, problems: Dict[str, Rule], inferencer: Inferencer,
                   rule_mapping: Dict, logger: logging.Logger = module_logger, **kwargs):
    problem_statistics = []
    problem_solutions = []

    show_progress = not logger.isEnabledFor(logging.DEBUG) and sys.stdin.isatty()

    context = Context.standard()

    def variable_generator():
        return Symbol.parse(context, 'u')

    eval_config = config.evaluation

    problem_traces = {}

    if options.smoke:
        problems = dict(itertools.islice(problems.items(), 1))

    for problem_name in tqdm(problems, desc='solve', disable=not show_progress, leave=False):
        problem = problems[problem_name]
        # Get first problem
        logger.debug(f'problem: {problem}')
        source = problem.condition
        target = problem.conclusion

        with Timer(f'Solving problem "{problem_name}"', logger=logger, quite=show_progress):
            search_strategy = beam_search_policy_last if options.policy_last else beam_search
            solution, statistics = search_strategy(inferencer, rule_mapping, problem.condition,
                                                   [problem.conclusion], variable_generator,
                                                   black_list_terms=eval_config.black_list_terms,
                                                   black_list_rules=eval_config.black_list_rules,
                                                   max_size=eval_config.max_size,
                                                   **vars(eval_config.problems), **kwargs)
        if solution is not None:
            problem_solutions.append(solution)

        else:
            logger.debug(f'No solution found for {source} => {target}')

        statistics.name = problem_name
        logger.debug(statistics)
        problem_statistics.append(statistics)
        problem_traces[problem_name] = statistics.as_builtin

    return problem_solutions, problem_statistics, problem_traces
