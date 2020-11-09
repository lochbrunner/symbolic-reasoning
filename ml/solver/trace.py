import logging
import yaml
from typing import List, Dict
from queue import Queue
from pathlib import Path

from dataset.bag import BagDataset

from pycore import Rule, FitInfo, SampleSet, Sample, Bag


class ApplyInfo:
    def __init__(self, rule_name: str, rule_formula: str, current, previous,
                 mapping, confidence, top: int, rule_id: int, path: list):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.value = None
        self.previous = previous
        self.subsequent: List[ApplyInfo] = []
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
        '''Rules from all previous steps to current'''
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
        if apply_info.previous:
            apply_info.previous.subsequent.append(apply_info)
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


class Tops:
    def __init__(self, N=8, prev=None):
        self.N = N
        if prev:
            self.values = {**prev}
        else:
            self.values = {}
        self.total = 0

    def __add__(self, index):
        if index not in self.values:
            self.values[index] = 1
        else:
            self.values[index] += 1
        self.total += 1
        return self

    @property
    def worst(self):
        return max(self.values.keys())


def calculate_policy_tops(solutions: List[ApplyInfo], prev_tops: Dict[int, int] = None):
    # # tops:
    # # tops begin with 1
    # if prev_tops:
    #     tops = {**prev_tops}
    # else:
    #     tops = {}
    # total = 0
    tops = Tops()
    for solution in solutions:
        for step in solution.trace:
            tops += step.top

    return tops


def calculate_value_tops(solutions: List[ApplyInfo]):
    tops = Tops()

    for solution in solutions:
        for step in solution.trace:
            if step.previous is None:
                continue
            if step.value is None:
                continue

            sisters = [s for s in step.previous.subsequent if s.value]
            sisters.sort(key=lambda info: info.value, reverse=True)
            self_index = next(i for i, sister in enumerate(sisters) if sister is step)
            tops += self_index

    return tops


def solution_summary(solutions: List[ApplyInfo], prev_tops: Dict[int, int] = None):

    return {
        'policy': calculate_policy_tops(solutions, prev_tops),
        'value': calculate_value_tops(solutions)
    }


class Statistics:
    '''Holds all relevant information of a solve try.'''

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

    def __init__(self, config):
        self.sample_set = SampleSet()
        self.solver_trainings_data = Path(config.evaluation.solver_trainings_data)
        self.initial_bag = Bag.load(str(config.files.trainings_data))

    def __add__(self, statistics: Statistics):
        for apply_info in statistics.trace.iter():
            if apply_info.rule_id is not None:
                sample = Sample(apply_info.previous.current, [apply_info.fit_info], apply_info.contributed)
                self.sample_set.add(sample)
        return self

    def dump(self):
        bag = self.initial_bag
        bag.clear_containers()
        bag.add_container(self.sample_set.to_container())
        bag.update_meta()
        self.solver_trainings_data.parent.mkdir(exist_ok=True, parents=True)
        logging.info(f'Dumping trainings data to {self.solver_trainings_data}')
        bag.dump(str(self.solver_trainings_data))

    def append(self):
        '''Appends itself to the solver trainings data if available. Else creates new file.'''

        if self.solver_trainings_data.exists():
            bag = Bag.load(str(self.solver_trainings_data))
            for container in bag.containers:
                self.sample_set.merge(SampleSet.from_container(container))
        else:
            bag = self.initial_bag
            bag.clear_containers()

        bag.add_container(self.sample_set.to_container())
        bag.update_meta()
        self.solver_trainings_data.parent.mkdir(exist_ok=True, parents=True)
        logging.info(f'Appending trainings data to {self.solver_trainings_data}')
        bag.dump(str(self.solver_trainings_data))

    def get_dataset(self):
        container = self.sample_set.to_container()
        return BagDataset(meta=self.initial_bag.meta, samples=container.samples,
                          max_depth=container.max_depth, max_size=container.max_size)


def dump_new_rules(solutions: List[ApplyInfo], new_rules_filename: Path, **kwargs):

    new_rules = [rule.verbose for solution in solutions
                 for rule in solution.new_rules()]
    new_rules_filename.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f'Dumping new rules to {new_rules_filename}')
    with new_rules_filename.open('w') as f:
        yaml.dump({
            'rules': new_rules
        }, f)
