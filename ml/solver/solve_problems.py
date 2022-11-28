from typing import Optional, Sequence
import itertools
import logging
import sys
import multiprocessing

from tqdm import tqdm

from pycore import Symbol, Context, Rule

from solver.trace import ApplyInfo, Statistics
from solver.beam_search import beam_search, beam_search_policy_last
from solver.inferencer import Inferencer


module_logger = logging.getLogger(__name__)


def search_fn(args: dict) -> tuple[Optional[ApplyInfo], Statistics, Rule]:
    problem = args['problem']
    context = Context.standard()

    def variable_generator():
        return Symbol.parse(context, 'u')

    return (
        *args['fn'](
            variable_generator=variable_generator,
            initial=problem.condition,
            targets=[problem.conclusion],
            **args,
        ),
        problem,
    )


def solve_problems(
    options,
    config,
    problems: Sequence[Rule],
    inferencer: Inferencer,
    rule_mapping: dict,
    logger: logging.Logger = module_logger,
    use_network: bool = True,
    **kwargs,
):

    show_progress = not logger.isEnabledFor(logging.DEBUG) and sys.stdin.isatty()

    eval_config = config.evaluation

    if options.smoke:
        problems = list(itertools.islice(problems, 1))

    if hasattr(eval_config.problems, 'white_list') and eval_config.problems.white_list:
        white_list = set(s.replace(' ', '') for s in eval_config.problems.white_list)
        unfiltered_problems = problems
        problems = [
            problem for problem in problems if str(problem.condition) in white_list
        ]
        if len(problems) < 1:
            print(
                'Available:\n'
                + '\n'.join(f'"{v.condition}"' for v in unfiltered_problems)
            )
            raise AssertionError(f'No problems found in the white list: {white_list}')
        logger.warning(f'Using {len(problems)} filtered problems from white list')

    search_strategy = beam_search_policy_last if options.policy_last else beam_search

    problem_args = [
        dict(
            inference=inferencer,
            rule_mapping=rule_mapping,
            use_network=use_network,
            black_list_terms=getattr(eval_config, 'black_list_terms', []),
            white_list_terms=getattr(eval_config, 'white_list_terms', []),
            black_list_rules=getattr(eval_config, 'black_list_rules', []),
            max_size=eval_config.max_size,
            max_grow=eval_config.max_grow,
            problem=problem,
            fn=search_strategy,
            **vars(eval_config.problems),
            **kwargs,
        )
        for problem in problems
    ]

    with multiprocessing.Pool(processes=8) as pool:
        pbar = tqdm(
            pool.imap_unordered(search_fn, problem_args),
            desc='solve (success: 0)',
            disable=not show_progress,
            leave=False,
            total=len(problems),
        )
        success_count = 0
        for solution, statistics, problem in pbar:
            # Get first problem
            logger.debug(f'problem: {problem}')

            if solution is None:
                logger.debug(
                    f'No solution found for {problem.condition} => {problem.conclusion}'
                )
            else:
                success_count += 1
                pbar.set_description(f'solve (success: {success_count})')

            statistics.name = problem.name
            logger.debug(statistics)

            yield statistics, solution
