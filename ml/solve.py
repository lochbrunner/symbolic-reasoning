#!/usr/bin/env python3

import logging
import yaml
from glob import glob

import numpy as np

import torch

from pycore import Symbol, Scenario, Trace, fit, apply, fit_at_and_apply

from common import io
from common.timer import Timer
from common.config_and_arg_parser import Parser as ArgumentParser

from dataset.transformers import Padder, Embedder, ident_to_id


class ApplyInfo:
    def __init__(self, rule_name, rule_formula, current, previous, mapping):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.previous = previous
        self.mapping = mapping

    def as_dict(self):
        return {
            'rule-name': self.rule_name,
            'current': self.current.latex,
            # 'previous': self.previous.current.latex if self.previous else None,
        }


class Statistics:
    def __init__(self, initial):
        self.name = ''
        self.initial_latex = initial.latex
        self.success = False
        self.fit_tries = 0
        self.fit_results = 0
        self.trace = LocalTrace(initial)

    def __str__(self):
        return f'Performing {self.fit_tries} fits results in {self.fit_results} fitting maps.'

    def as_dict(self):
        return {
            'name': self.name,
            'initial_latex': self.initial_latex,
            'trace': self.trace.as_dict(),
            'success': self.success,
            'fit_tries': self.fit_tries,
            'fit_results': self.fit_results,
        }


class Inferencer:
    def __init__(self, model_filename, depth=None):
        self.model, self.idents, _ = io.load_model(model_filename, depth=depth)
        self.model.eval()
        self.spread = self.model.spread
        self.depth = self.model.depth

        self.paths = list(Embedder.legend(spread=self.spread, depth=self.depth)) + [None]

    def create_mask(self, node):
        def hash_path(path):
            if path is None:
                return '<padding>'
            return '/'.join([str(p) for p in path])
        path_ids = set()
        for path, _ in node.parts_dfs_with_path:
            path_ids.add(hash_path(path))

        return np.array([(hash_path(p) in path_ids) for p in self.paths])

    def __call__(self, node, count):
        assert self.depth >= node.depth, f'{self.depth} >= {node.depth}'
        x = Padder.pad(node, spread=self.spread, depth=self.depth)
        x = [ident_to_id(n, self.idents) for n in Embedder.unroll(x)] + [0]
        x = torch.as_tensor(x, dtype=torch.long, device=self.model.device)
        y = self.model(x.unsqueeze(0))

        y = y.squeeze()
        mask = self.create_mask(node)
        y = y.cpu().detach().numpy()[1:]  # Remove padding
        y = y[:, mask]
        paths = [path for i, path in enumerate(self.paths) if mask[i]]
        i = (-y).flatten().argsort()

        def calc(n):
            p = np.unravel_index(i[n], y.shape)
            return p[0]+1, paths[p[1]]  # rule at path

        return [calc(i) for i in range(count)]


class LocalTrace:
    class Node:
        def __init__(self, apply_info):
            self.apply_info = apply_info
            self.childs = []

    def __init__(self, initial):
        self.root = LocalTrace.Node(ApplyInfo(
            rule_name='initial', rule_formula='',
            current=initial, previous=None, mapping=None))
        self.current_stage = [self.root]
        self.current_index = None
        self.next_stage = []

    def __len__(self):
        return len(self.current_stage)

    def __getitem__(self, index):
        self.current_index = index
        return self.current_stage[index].apply_info

    def close_stage(self):
        self.current_stage = self.next_stage
        self.next_stage = []
        self.current_index = None

    def add(self, apply_info):
        node = LocalTrace.Node(apply_info)
        self.current_stage[self.current_index].childs.append(node)
        self.next_stage.append(node)

    def as_dict_recursive(self, node):
        return {'apply_info': node.apply_info.as_dict(),
                'childs': [self.as_dict_recursive(c) for c in node.childs]}

    def as_dict(self):
        return self.as_dict_recursive(self.root)


def beam_search(inference, rule_mapping, initial, targets, variable_generator, beam_size, num_epochs, **kwargs):
    seen = set()
    statistics = Statistics(initial)

    for _ in range(num_epochs):
        for prev in statistics.trace:
            policies = inference(prev.current, beam_size)
            for (rule_id, path) in policies:
                rule = rule_mapping[rule_id-1]
                result = fit_at_and_apply(variable_generator, prev.current, rule, path)
                statistics.fit_tries += 1
                statistics.fit_results += 1 if result is not None else 0
                if result is None:
                    logging.debug(f'Missing fit of {rule.condition} at {path} in {prev.current}')
                    continue
                deduced, mapping = result
                s = str(deduced)
                if s in seen:
                    continue
                seen.add(s)
                apply_info = ApplyInfo(
                    rule_name=rule.name, rule_formula=str(rule),
                    current=deduced,
                    previous=prev, mapping=mapping)
                statistics.trace.add(apply_info)
                if deduced in targets:
                    statistics.success = True
                    return apply_info, statistics

        statistics.trace.close_stage()

    return None, statistics


def solve_training_problems(training_traces, scenario, model, rule_mapping, **kwargs):

    def variable_generator():
        return Symbol.parse(scenario.declarations, 'u')

    inferencer = Inferencer(model, depth=9)

    failed = 0
    succeeded = 0
    for filename in glob(training_traces):
        logging.info(f'Evaluating {filename} ...')
        trace = Trace.load(filename)
        for calculation in trace.unroll:
            conclusion = calculation.steps[0].initial
            condition = calculation.steps[-1].initial
            solution, _ = beam_search(inferencer, rule_mapping, condition,
                                      [conclusion], variable_generator, **kwargs)
            if solution is not None:
                succeeded += 1
            else:
                failed += 1
            logging.info(f'{succeeded} succeeded out of {succeeded+failed}')

    logging.info(f'{succeeded} of {succeeded+failed} training traces succeeded')
    return {'succeeded': succeeded, 'failed': failed}


def main(scenario, model, results_filename, training_traces, solve_training, **kwargs):
    with Timer('Loading model'):
        scenario = Scenario.load(scenario)

    # model, idents, rules = io.load_model(model, depth=9)
    rules = io.load_rules(model)
    rule_mapping = {}
    used_rules = set()
    for i, model_rule in enumerate(rules[1:]):
        scenario_rule = next(rule for rule in scenario.rules.values() if str(rule.reverse) == model_rule)
        rule_mapping[i] = scenario_rule
        used_rules.add(str(scenario_rule))
        logging.debug(f'Using rule {i}# {scenario_rule}')

    for scenario_rule in scenario.rules.values():
        if str(scenario_rule) not in used_rules:
            logging.warning(f'{scenario_rule} was not in the training of the model')

    if solve_training:
        training_statistics = solve_training_problems(
            training_traces, scenario, model, rule_mapping, **kwargs)
    else:
        training_statistics = {}

    problem_statistics = []

    def variable_generator():
        return Symbol.parse(context, 'u')
    context = scenario.declarations
    for problem_name in scenario.problems:
        problem = scenario.problems[problem_name]
        # Get first problem
        logging.info(f'problem: {problem}')
        logging.info('-'*30)
        source = problem.condition
        target = problem.conclusion

        with Timer('Solving problem'):
            solution, statistic = beam_search(Inferencer(model), rule_mapping, problem.condition,
                                              [problem.conclusion], variable_generator, **kwargs)

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

        statistic.name = problem_name
        logging.info(statistic)
        problem_statistics.append(statistic.as_dict())

    with open(results_filename, 'w') as f:
        yaml.dump({'problems': problem_statistics,
                   'training-traces': training_statistics}, f)


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return {'model': config['files']['model'],
                'results-filename': config['files']['evaluation-results'],
                'training-traces': config['files']['trainings-data-traces'],
                **config['evaluation']
                }


if __name__ == '__main__':
    parser = ArgumentParser('-c', '--config-file', config_name='scenario', loader=load_config,
                            prog='solver')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--beam-size', default=10)
    parser.add_argument('--results-filename')
    parser.add_argument('--training-traces')
    # Model
    parser.add_argument('-m', '--model', help='Filename of the model snapshot')
    args = parser.parse_args()

    args = parser.parse_args()
    loglevel = 'DEBUG' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )
    main(**vars(args))
