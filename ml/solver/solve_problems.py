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
                   rule_mapping: Dict, logger: logging.Logger = module_logger, use_network: bool = True, **kwargs):
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

    if hasattr(eval_config.problems, 'white_list') and eval_config.problems.white_list:
        white_list = set(s.replace(' ', '') for s in eval_config.problems.white_list)
        unfiltered_problems = problems
        problems = {n: v for n, v in problems.items() if str(v.condition) in white_list}
        if len(problems) < 1:
            print('Available:\n' + '\n'.join(f'"{v.condition}"' for _, v in unfiltered_problems.items()))
            raise AssertionError(f'No problems found in the white list: {white_list}')
        logger.warning(f'Using {len(problems)} filtered problems from white list')

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
                                                   use_network=use_network,
                                                   black_list_terms=eval_config.black_list_terms,
                                                   black_list_rules=eval_config.black_list_rules,
                                                   max_size=eval_config.max_size,
                                                   max_grow=eval_config.max_grow,
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
