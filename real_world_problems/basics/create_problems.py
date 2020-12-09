#!/usr/bin/env python3


import solving_problems
from pathlib import Path
import yaml
import random
from tqdm import tqdm

from pycore import ScenarioProblems, Rule, Context


def main():

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

    problems = solving_problems.create_linear(3)
    problem_size = len(problems)

    # # Take 10% as validation set
    validation_indices = set(random.sample(range(problem_size), k=problem_size // 10))

    for i, problem in tqdm(enumerate(problems), desc=f'Dumping to {problems_path}', smoothing=0.):
        rule = Rule.parse(context, problem)
        name = f'solving {i+1}'
        if i in validation_indices:
            scenario_problems.add_to_validation(rule, name)
        else:
            scenario_problems.add_to_training(rule, name)

    scenario_problems.dump(problems_path)


if __name__ == '__main__':
    main()
