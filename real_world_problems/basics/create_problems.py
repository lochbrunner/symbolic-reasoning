#!/usr/bin/env python3


import solving_problems
from pathlib import Path
import yaml
import random
from tqdm import tqdm
import logging
import argparse

from pycore import ScenarioProblems, Rule, Context

logger = logging.getLogger(__name__)


def main(args):
    loglevel = 'INFO' if args.verbose else args.log.upper()
    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s',
        datefmt='%I:%M:%S'
    )

    random.seed(0)

    me = Path(__file__).absolute()
    root = me.parents[2]

    config_path = me.parent / 'config.yaml'

    with config_path.open() as f:
        config = yaml.safe_load(f)

        scenario_path = root / config['files']['scenario']
    with scenario_path.open() as f:
        scenario_data = yaml.safe_load(f)

        problems_path = scenario_data['problems']['filename']

    scenario_problems = ScenarioProblems()
    context = Context.standard()

    problems, used_idents = solving_problems.create_linear(**vars(args))
    problem_size = len(problems)

    # # Take 10% as validation set
    validation_indices = set(random.sample(range(problem_size), k=problem_size // 10))

    for i, problem in tqdm(enumerate(problems), desc=f'Dumping to {problems_path}', smoothing=0., leave=False):
        rule = Rule.parse(context, problem)
        name = f'solving {i+1}'
        if i in validation_indices:
            scenario_problems.add_to_validation(rule, name)
        else:
            scenario_problems.add_to_training(rule, name)

    scenario_problems.dump(problems_path)
    logger.info(f'used_idents: {", ".join(used_idents)}')
    scenario_problems.add_additional_idents(list(set(used_idents)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Problems creator')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--left-length', default=3, type=int)
    parser.add_argument('--right-length', default=3, type=int)
    parser.add_argument('--num-symbols', default=2, type=int)
    main(parser.parse_args())
