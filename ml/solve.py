#!/usr/bin/env python3

import argparse
import logging

from pycore import Symbol, Scenario, fit, apply

from common import io


class ApplyInfo:
    def __init__(self, rule_name, rule_formula, current, previous, mapping):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.previous = previous
        self.mapping = mapping


def main(scenario, model, **kwargs):
    scenario = Scenario.load(scenario)
    # model = io.load(model)

    # Get first problem
    problem = scenario.problems['two steps']
    print(f'problem: {problem}')
    print('-'*30)
    source = problem.condition
    target = problem.conclusion
    context = scenario.declarations

    def variable_generator():
        return Symbol.parse(context, 'u')

    rules = scenario.rules

    traces = [ApplyInfo(
        rule_name='initial', rule_formula='',
        current=problem.condition, previous=None, mapping=None)]

    seen = set()
    num_epochs = 2
    solution = None
    for _ in range(num_epochs):
        prevs = traces.copy()
        traces.clear()
        for prev in prevs:
            for rule in rules.values():
                mappings = fit(prev.current, rule.condition)
                for mapping in mappings:
                    deduced = apply(mapping, variable_generator, prev.current, rule.conclusion)
                    s = str(deduced)
                    if s in seen:
                        continue
                    seen.add(s)
                    apply_info = ApplyInfo(
                        rule_name=rule.name, rule_formula=str(rule),
                        current=deduced,
                        previous=prev, mapping=mapping)
                    if deduced == target:
                        solution = apply_info
                    else:
                        traces.append(apply_info)

    if solution is not None:
        trace = []
        while solution is not None:
            trace.append(solution)
            solution = solution.previous

        for step in reversed(trace):
            if step.previous is not None:
                print(f'{step.rule_name} ({step.rule_formula}): {step.previous.current} => {step.current} ({step.mapping})')
            else:
                print(f'Initial: {step.current}')
    else:
        logging.warning(f'No solution found for {source} => {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('solver')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    # Scenario
    parser.add_argument('-s', '--scenario', help='Filename of the scenario')
    # Model
    parser.add_argument('-m', '--model', help='Filename of the model snapshot')
    args = parser.parse_args()

    args = parser.parse_args()
    loglevel = 'INFO' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )
    main(**vars(args))
