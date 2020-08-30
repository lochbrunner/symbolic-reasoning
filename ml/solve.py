#!/usr/bin/env python3

import logging
import yaml
from glob import glob
from time import time
from pathlib import Path
from queue import Queue

from typing import List

import numpy as np

import torch

from pycore import Symbol, Scenario, Trace, fit, apply, fit_at_and_apply, fit_and_apply, Bag, FitInfo, Sample, Container

from common import io
from common.timer import Timer
from common.config_and_arg_parser import Parser as ArgumentParser

from dataset.transformers import Padder, Embedder, ident_to_id


class ApplyInfo:
    def __init__(self, rule_name: str, rule_formula, current, previous,
                 mapping, confidence, top: int, rule_id: int, path: list):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.previous = previous
        self.mapping = mapping
        self.rule_id = rule_id
        self.path = path
        self.top = top
        self.contributed = False
        if hasattr(confidence, 'item'):
            self.confidence = confidence.item()
        else:
            self.confidence = confidence

    def as_dict(self):
        return {
            'rule-name': self.rule_name,
            'current': self.current.latex_verbose,
            'confidence': self.confidence,
            'top': self.top,
            'contributed': self.contributed,
            'rule_id': self.rule_id,
        }

    def contribute(self):
        self.contributed = True
        if self.previous is not None:
            self.previous.contribute()

    @property
    def fit_info(self):
        return FitInfo(self.rule_id, self.path, self.contributed)

    @property
    def trace(self):
        step = self
        while step is not None:
            yield step
            step = step.previous


class Statistics:
    def __init__(self, initial):
        self.name = ''
        self.initial_latex = initial.latex_verbose
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


def solution_summary(solutions: List[ApplyInfo]):
    # tops:
    # tops begin with 1
    tops = {}
    total = 0
    for solution in solutions:
        for step in solution.trace:
            if step.top in tops:
                tops[step.top] += 1
            else:
                tops[step.top] = 1
        total += 1
    tops['total'] = total
    return {'tops': tops}


class Inferencer:
    ''' Standard inferencer for unique index map per sample
    '''

    def __init__(self, model_filename):
        self.model, snapshot = io.load_model(model_filename)
        self.model.eval()
        # Copy of BagDataset
        self.ident_dict = {ident: (value+1) for (value, ident) in enumerate(snapshot['idents'])}
        self.spread = snapshot['kernel_size'] - 2
        self.pad_token = snapshot['pad_token']

    def __call__(self, initial, count=None):
        # x, s, _ = self.dataset.embed_custom(initial)
        x, s, _, _ = initial.embed(self.ident_dict, self.pad_token, self.spread, [])
        x = torch.unsqueeze(torch.as_tensor(np.copy(x), device=self.model.device), 0)
        s = torch.unsqueeze(torch.as_tensor(np.copy(s), device=self.model.device), 0)
        p = torch.ones(x.shape[:-1])

        y = self.model(x, s, p)
        y = y.squeeze()  # shape: rules, localisation
        y = y.cpu().detach().numpy()[1:, :-1]  # Remove padding

        parts_path = [p[0] for p in initial.parts_bfs_with_path]
        i = (-y).flatten().argsort()

        def calc(n):
            p = np.unravel_index(i[n], y.shape)
            return p[0]+1, parts_path[p[1]], y[p[0], p[1]]  # rule at path

        if count is None:
            count = i.shape[0]

        return [calc(i) for i in range(count)]


class SharedInferencer:
    '''Deprecated'''

    def __init__(self, model_filename, depth=None):
        self.model, snapshot = io.load_model(model_filename, depth=depth)
        self.idents = snapshot['idents']
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
            current=initial, previous=None, mapping=None,
            confidence=1, top=1, rule_id=None, path=None))
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

    def iter(self):
        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            for child in node.childs:
                queue.put(child)
            yield node.apply_info

    def as_dict(self):
        return self.as_dict_recursive(self.root)


def beam_search(inference, rule_mapping, initial, targets, variable_generator, beam_size, num_epochs, **kwargs):
    '''First apply the policy and then try to fit the suggestions.'''
    seen = set()
    statistics = Statistics(initial)

    for _ in range(num_epochs):
        for prev in statistics.trace:
            policies = inference(prev.current, beam_size)
            for top, (rule_id, path, confidence) in enumerate(policies, 1):
                rule = rule_mapping[rule_id-1]
                result = fit_at_and_apply(variable_generator, prev.current, rule, path)
                statistics.fit_tries += 1
                statistics.fit_results += 1 if result is not None else 0
                if result is None:
                    logging.debug(f'Missing fit of {rule.condition} at {path} in {prev.current}')
                    continue
                deduced, mapping = result
                s = deduced.verbose
                if s in seen:
                    continue
                seen.add(s)
                apply_info = ApplyInfo(
                    rule_name=rule.name, rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev, mapping=mapping,
                    confidence=confidence,
                    top=top,
                    rule_id=rule_id, path=path)
                statistics.trace.add(apply_info)
                if deduced in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics

        statistics.trace.close_stage()

    return None, statistics


def beam_search_policy_last(inference, rule_mapping, initial, targets, variable_generator, beam_size, num_epochs, black_list_terms, black_list_rules, max_size, **kwargs):
    '''Same as `beam_search` but first get fit results and then apply policy to sort the results.'''
    black_list_terms = set(black_list_terms)
    black_list_rules = set(black_list_rules)
    seen = set([initial.verbose])
    statistics = Statistics(initial)
    for epoch in range(num_epochs):
        logging.debug(f'epoch: {epoch}')
        successfull_epoch = False
        for prev in statistics.trace:
            possible_rules = {}
            for i, rule in rule_mapping.items():
                if rule.name not in black_list_rules:
                    if fits := fit_and_apply(variable_generator, prev.current, rule):
                        possible_rules[i] = fits

            # Sort the possible fits by the policy network
            policies = inference(prev.current, None)  # rule_id, path
            ranked_fits = {}
            for rule_id, fits in possible_rules.items():
                for deduced, fit_result in fits:
                    try:
                        j, confidence = next((i, conf) for i, (pr, pp, conf) in enumerate(policies)
                                             if pr == rule_id and pp == fit_result.path)
                    except StopIteration:
                        for k, v in rule_mapping.items():
                            print(f'#{k}: {v}')
                        raise RuntimeError(f'Can not find {rule_mapping[rule_id]} #{rule_id} at {fit_result.path}')
                    # rule id, path, mapping, deduced
                    ranked_fits[j] = (rule_id, fit_result, confidence, deduced)

            possible_fits = (v for _, v in sorted(ranked_fits.items()))

            # filter out already seen terms
            possible_fits = ((*args, deduced) for *args, deduced in possible_fits if deduced.verbose
                             not in seen and deduced.verbose not in black_list_terms)

            for top, (rule_id, fit_result, confidence, deduced) in enumerate(possible_fits, 1):
                seen.add(deduced.verbose)
                if deduced.size > max_size:
                    continue
                rule = rule_mapping[rule_id]
                # print(deduced.verbose)
                apply_info = ApplyInfo(
                    rule_name=rule.name, rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev, mapping=fit_result.variable,
                    confidence=confidence, top=top, rule_id=rule_id, path=fit_result.path)
                statistics.trace.add(apply_info)
                successfull_epoch = True

                if deduced in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics
        if not successfull_epoch:
            break
        statistics.trace.close_stage()

    return None, statistics


def solve_training_problems(training_traces, scenario, model, rule_mapping, training_data_max_steps, **kwargs):

    def variable_generator():
        return Symbol.parse(scenario.declarations, 'u')

    inferencer = Inferencer(model)

    failed = 0
    succeeded = 0
    total_duration = 0.
    seen = set()
    for filename in glob(training_traces):
        logging.info(f'Evaluating {filename} ...')
        trace = Trace.load(filename)
        for calculation in trace.unroll:
            conclusion = calculation.steps[0].initial
            steps_count = min(training_data_max_steps, len(calculation.steps)-1)
            condition = calculation.steps[steps_count].initial
            if condition in seen:
                continue
            seen.add(condition)
            start_time = time()
            solution, _ = beam_search(inferencer, rule_mapping, condition,
                                      [conclusion], variable_generator, **kwargs)
            total_duration += time() - start_time
            if solution is not None:
                succeeded += 1
            else:
                failed += 1

    logging.info(f'{succeeded} of {succeeded+failed} training traces succeeded')
    total = succeeded + failed
    return {'succeeded': succeeded, 'failed': failed, 'total': total, 'mean-duration': total_duration / total}


def dump_trainings_data(statistics: Statistics, solver_trainings_data: str, initial_trainings_data_file: str, **kwargs):

    container = Container()
    for apply_info in statistics.trace.iter():
        # For now each fit it's own sample
        if apply_info.rule_id is not None:
            sample = Sample(apply_info.previous.current, [apply_info.fit_info])
            container.add_sample(sample)

    bag = Bag.load(initial_trainings_data_file)
    bag.clear_containers()
    bag.add_container(container)
    bag.update_meta()
    Path(solver_trainings_data).parent.mkdir(exist_ok=True, parents=True)
    logging.info(f'Dumping trainings data to {solver_trainings_data}')
    bag.dump(solver_trainings_data)


def main(scenario, model, results_filename, training_traces, solve_training, problems_beam_size, training_data_beam_size, policy_last, **kwargs):
    with Timer('Loading model'):
        scenario = Scenario.load(scenario)

    # model, idents, rules = io.load_model(model, depth=9)
    rules = io.load_rules(model)
    rule_mapping = {}
    used_rules = set()
    max_width = max(len(s.name) for s in scenario.rules.values())+1
    for i, model_rule in enumerate(rules[1:], 1):
        scenario_rule = next(rule for rule in scenario.rules.values() if rule.reverse.verbose == model_rule)
        rule_mapping[i] = scenario_rule
        used_rules.add(str(scenario_rule))
        logging.debug(f'Using rule {i:2}# {scenario_rule.name.ljust(max_width)} {scenario_rule.verbose}')

    for scenario_rule in scenario.rules.values():
        if str(scenario_rule) not in used_rules:
            logging.warning(f'The rule "{scenario_rule}" was not in the model created by the training.')

    training_statistics = []
    if solve_training:
        bs = [int(s) for s in str(training_data_beam_size).split(':')]
        if len(bs) == 1:
            beam_size_range = bs
        else:
            beam_size_range = range(*bs)
        for bs in beam_size_range:
            logging.info(f'Using beam size {bs}')
            training_statistic = solve_training_problems(
                training_traces, scenario, model, rule_mapping, beam_size=bs, **kwargs)
            training_statistic['beam-size'] = bs
            training_statistics.append(training_statistic)

    problem_statistics = []
    problem_solutions = []

    def variable_generator():
        return Symbol.parse(context, 'u')
    context = scenario.declarations
    inferencer = Inferencer(model)
    for problem_name in scenario.problems:
        problem = scenario.problems[problem_name]
        # Get first problem
        logging.info(f'problem: {problem}')
        logging.info('-'*30)
        source = problem.condition
        target = problem.conclusion

        with Timer(f'Solving problem "{problem_name}"'):
            search_strategy = beam_search_policy_last if policy_last else beam_search
            solution, statistics = search_strategy(inferencer, rule_mapping, problem.condition,
                                                   [problem.conclusion], variable_generator, beam_size=problems_beam_size, **kwargs)

        if solution is not None:
            problem_solutions.append(solution)
            for step in reversed(list(solution.trace)):
                if step.previous is not None:
                    mapping = ', '.join([f'{a} -> {b}' for a, b in step.mapping.items()])
                    if mapping:
                        mapping = f' ({mapping})'
                    print(f'{step.rule_name} ({step.rule_formula}): {step.previous.current.verbose} => {step.current.verbose}{mapping}')
                else:
                    print(f'Initial: {step.current}')

            dump_trainings_data(statistics, rules=scenario.rules, **kwargs)
        else:
            logging.warning(f'No solution found for {source} => {target}')

        statistics.name = problem_name
        logging.info(statistics)
        problem_statistics.append(statistics.as_dict())

    logging.info(f'Writing results to {results_filename}')
    with open(results_filename, 'w') as f:
        yaml.dump({
            'problems': problem_statistics,
            'problem-statistics': solution_summary(problem_solutions),
            'training-traces': training_statistics
        }, f)


def load_config(filename):
    def unroll_nested_dict(dictionary, parent=''):
        for key, value in dictionary.items():
            if type(value) in (str, int, float, list):
                yield (f'{parent}{key}', value)
            elif type(value) is dict:
                for pair in unroll_nested_dict(value, f'{parent}{key}-'):
                    yield pair
            else:
                raise NotImplementedError(f'Loading config does not support type {type(value)}')

    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return {'model': config['files']['model'],
                'results-filename': config['files']['evaluation-results'],
                'training-traces': config['files']['trainings-data-traces'],
                'initial-trainings-data-file': config['files']['trainings-data'],
                **{k: v for k, v in unroll_nested_dict(config['evaluation'])}
                }


if __name__ == '__main__':
    parser = ArgumentParser('-c', '--config-file', config_name='scenario', loader=load_config,
                            prog='solver')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--problems-beam-size', default=10)
    parser.add_argument('--results-filename')
    parser.add_argument('--training-traces')
    parser.add_argument('--training-data-max-steps')
    parser.add_argument('--training-data-beam-size')
    parser.add_argument('--solver-trainings-data')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--black-list-terms', nargs='+')
    parser.add_argument('--black-list-rules', nargs='+')
    parser.add_argument('--max-size', type=int)
    parser.add_argument('--policy-last', action='store_true', default=False)

    # Model
    parser.add_argument('-m', '--model', help='Filename of the model snapshot')
    args = parser.parse_args()

    args, _ = parser.parse_args()
    loglevel = 'DEBUG' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )
    main(**vars(args))
