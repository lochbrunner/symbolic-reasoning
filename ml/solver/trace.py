
import logging
import yaml
from typing import List
from queue import Queue
from pathlib import Path

from pycore import Rule, FitInfo, SampleSet, Sample, Bag


class ApplyInfo:
    def __init__(self, rule_name: str, rule_formula, current, previous,
                 mapping, confidence, top: int, rule_id: int, path: list):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.value = None
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
            'value': self.value,
            'contributed': self.contributed,
            'rule_id': self.rule_id,
        }

    def contribute(self):
        self.contributed = True
        if self.previous is not None:
            self.previous.contribute()

    def new_rules(self):
        '''Rules from all previous step to current'''
        for i, step in enumerate(self.trace, 1):
            if step.previous is not None:
                yield Rule(condition=step.previous.current, conclusion=self.current, name=f'New rule {i}')

    @property
    def fit_info(self):
        return FitInfo(self.rule_id, self.path, self.contributed)

    @property
    def trace(self):
        step = self
        while step is not None:
            yield step
            step = step.previous


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
        self.size = 0

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
        self.size += 1

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


class TrainingsDataDumper:
    '''Stores all samples and dumps them on demand.'''

    def __init__(self, solver_trainings_data: Path, initial_trainings_data_file: str, **kwargs):
        self.sample_set = SampleSet()
        self.solver_trainings_data = solver_trainings_data
        self.initial_trainings_data_file = initial_trainings_data_file

    def __add__(self, statistics: Statistics):
        for apply_info in statistics.trace.iter():
            if apply_info.rule_id is not None:
                sample = Sample(apply_info.previous.current, [apply_info.fit_info], apply_info.contributed)
                self.sample_set.add(sample)
        return self

    def dump(self):
        bag = Bag.load(self.initial_trainings_data_file)
        bag.clear_containers()
        bag.add_container(self.sample_set.to_container())
        bag.update_meta()
        self.solver_trainings_data.parent.mkdir(exist_ok=True, parents=True)
        logging.info(f'Dumping trainings data to {self.solver_trainings_data}')
        bag.dump(str(self.solver_trainings_data))


def dump_new_rules(solutions: List[ApplyInfo], new_rules_filename: Path, **kwargs):

    new_rules = [rule.verbose for solution in solutions
                 for rule in solution.new_rules()]
    new_rules_filename.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f'Dumping new rules to {new_rules_filename}')
    with new_rules_filename.open('w') as f:
        yaml.dump({
            'rules': new_rules
        }, f)
