import logging
import yaml
from typing import Sequence, Optional
from queue import Queue
import pathlib

from dataset.bag import BagDataset

Path = Sequence[int]

from pycore import (
    Rule,
    FitInfo,
    SampleSet,
    Sample,
    Bag,
    Scenario,
    BagMeta,
    StepInfo,
    TraceStatistics,
)

logger = logging.getLogger(__name__)


class ApplyInfo:
    def __init__(
        self,
        rule_name: str,
        rule_formula: str,
        current,
        previous,
        mapping,
        confidence,
        top: int,
        rule_id: Optional[int],  # None means initial
        path: Optional[Path],  # None means initial
    ):
        self.rule_name = rule_name
        self.rule_formula = rule_formula
        self.current = current
        self.value: Optional[float] = None
        self.previous = previous
        self.subsequent: Sequence[ApplyInfo] = []
        self.mapping = mapping
        self.rule_id = rule_id
        self.path = path
        self.top = top
        self.contributed = False
        self.alternative_traces = []

        if hasattr(confidence, 'item'):
            self.confidence = confidence.item()
        else:
            self.confidence = confidence

    @property
    def as_builtin(self) -> StepInfo:
        step = StepInfo()
        step.current_latex = self.current.latex_verbose
        if self.value is not None:
            step.value = self.value
        if self.confidence is not None:
            step.confidence = self.confidence
        if self.rule_id is None or self.path is None:
            raise AssertionError()
        step.rule_id = self.rule_id
        step.path = self.path
        step.top = self.top
        step.contributed = self.contributed

        for subsequent in self.subsequent:
            step.add_subsequent(subsequent.as_builtin)
        return step

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
        # Already done?
        if self.contributed:
            return
        self.contributed = True
        if self.previous is not None:
            self.previous.contribute()
        for alternative in self.alternative_traces:
            alternative.contribute()

    def new_rules(self):
        '''Rules from all previous steps to current'''
        for i, step in enumerate(self.trace, 1):
            if step.previous is not None:
                yield Rule(
                    condition=step.previous.current,
                    conclusion=self.current,
                    name=f'New rule {i}',
                )

    @property
    def fit_info(self):
        if self.rule_id is None or self.path is None:
            return None
        return FitInfo(self.rule_id, self.path, self.contributed)

    @property
    def trace(self):
        step = self
        while step is not None:
            yield step
            step = step.previous

    @property
    def track_loss(self):
        '''Number of steps to the last good value'''

        i = 0
        for i, step in enumerate(self.trace):
            if step.value is not None and step.value < 0.5:
                break

        return i


class LocalTrace:
    class Node:
        def __init__(self, apply_info: ApplyInfo):
            self.apply_info = apply_info
            self.childs = []

    def __init__(self, initial):
        self.root = LocalTrace.Node(
            ApplyInfo(
                rule_name='initial',
                rule_formula='',
                current=initial,
                previous=None,
                mapping=None,
                confidence=1,
                top=1,
                rule_id=None,
                path=None,
            )
        )
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
        if self.current_index is None:
            raise AssertionError()
        self.current_stage[self.current_index].childs.append(node)
        self.next_stage.append(node)
        self.size += 1

    @staticmethod
    def as_dict_recursive(node):
        return {
            'apply_info': node.apply_info.as_dict(),
            'childs': [LocalTrace.as_dict_recursive(c) for c in node.childs],
        }

    def iter(self, just_neighborhood=False):
        '''Traverses breath first through all nodes in the tree.'''

        if just_neighborhood and not self.root.apply_info.contributed:
            yield from ()

        queue = Queue()
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            # Look for previous in order to get negative value examples
            if (
                not just_neighborhood
                or not node.apply_info.previous
                or node.apply_info.previous.contributed
            ):
                for child in node.childs:
                    queue.put(child)
            yield node.apply_info

    def as_dict(self):
        return self.as_dict_recursive(self.root)

    @staticmethod
    def as_built_recursive(node: Node) -> StepInfo:
        step = StepInfo()
        source = node.apply_info
        step.current_latex = source.current.latex_verbose
        if source.value is not None:
            step.value = source.value
        if source.confidence is not None:
            step.confidence = source.confidence
        step.rule_id = source.rule_id or 0
        step.path = source.path or []
        step.top = source.top
        step.contributed = source.contributed

        for subsequent in node.childs:
            step.add_subsequent(LocalTrace.as_built_recursive(subsequent))
        return step

    @property
    def as_builtin(self) -> StepInfo:
        return self.as_built_recursive(self.root)


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

    @property
    def as_builtin(self) -> TraceStatistics:
        trace = TraceStatistics()
        trace.success = self.success
        trace.fit_tries = self.fit_tries
        trace.fit_results = self.fit_results
        trace.trace = self.trace.as_builtin

        return trace


class TrainingsDataDumper:
    '''Stores all samples and dumps them on demand.'''

    def __init__(self, config, scenario: Scenario):
        self.sample_set = SampleSet()
        self.solver_trainings_data = pathlib.Path(
            config.evaluation.solver_trainings_data
        )
        self.scenario = scenario

    def __add__(self, statistics: Statistics):
        if statistics.success:
            for apply_info in statistics.trace.iter(just_neighborhood=True):
                if apply_info.rule_id is not None:
                    # Just store fits of contributed steps
                    fits = (
                        [apply_info.fit_info] if apply_info.previous.contributed else []
                    )
                    sample = Sample(
                        apply_info.previous.current,
                        fits,
                        apply_info.previous.contributed,
                    )
                    self.sample_set.add(sample)
        return self

    def dump(self, rule_mapping: dict[int, Rule]):
        bag = Bag.from_scenario(self.scenario)
        self.sample_set.fill_possibilities(rule_mapping)
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
            bag = Bag.from_scenario(self.scenario)

        bag.add_container(self.sample_set.to_container())
        bag.update_meta()
        self.solver_trainings_data.parent.mkdir(exist_ok=True, parents=True)
        logging.info(f'Appending trainings data to {self.solver_trainings_data}')
        bag.dump(str(self.solver_trainings_data))

    def get_dataset(self):
        container = self.sample_set.to_container()
        meta = BagMeta.from_scenario(self.scenario)
        meta = meta.clone_with_distribution(container.samples)
        return BagDataset(
            meta=meta,
            samples=container.samples,
            max_depth=container.max_depth,
            max_size=container.max_size,
        )


def dump_new_rules(
    solutions: Sequence[ApplyInfo], new_rules_filename: pathlib.Path, **kwargs
):

    new_rules = [
        rule.verbose for solution in solutions for rule in solution.new_rules()
    ]
    new_rules_filename.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f'Dumping new rules to {new_rules_filename}')
    with new_rules_filename.open('w') as f:
        yaml.dump({'rules': new_rules}, f)
