from __future__ import annotations
from absl import logging
import yaml
from typing import Iterable, Iterator, Optional, Sequence, TypeVar
from queue import Queue
import pathlib
import unittest
import dataclasses

from dataset.bag import BagDataset
from pycore import FitMap, Symbol

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

_T = TypeVar('_T')


@dataclasses.dataclass
class ApplyInfo:
    rule_name: str
    rule_formula: str
    current: Symbol
    previous: Optional[ApplyInfo]  # None for roots
    mapping: Optional[FitMap]  # None means initial
    rule_id: Optional[int]  # None means initial
    path: Optional[Path]  # None means initial
    top: int
    confidence: Optional[float]
    value: Optional[float] = None
    subsequent: Sequence[ApplyInfo] = dataclasses.field(
        default_factory=list, init=False
    )
    contributed: bool = dataclasses.field(default=False, init=False)
    alternative_traces: Sequence[ApplyInfo] = dataclasses.field(
        default_factory=list, init=False
    )

    @classmethod
    def create_root(cls: type[_T], initial: Symbol) -> _T:
        return cls(
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

    @classmethod
    def create_for_test(
        cls: type[_T],
        rule_name: str,
        current: Symbol,
        previous: Optional[ApplyInfo] = None,  # None for roots
    ) -> _T:
        return cls(
            rule_name=rule_name,
            rule_formula=f'${rule_name}$',
            current=current,
            previous=previous,
            mapping=None,
            confidence=1.0,
            top=0,
            rule_id=0,
            path=None,
        )

    def _reverse_rule_formula(self, formula):
        terms = formula.split('=>')
        if len(terms) == 2:
            return f'{terms[1]}=>{terms[0]}'
        else:
            return f'reversed "{formula}"'

    def clone(self) -> ApplyInfo:
        return dataclasses.replace(self)

    def reversed(self, counter_apply_info: ApplyInfo) -> ApplyInfo:
        """Creates a reversed instance with counter_apply_info is the nre previous
        counter_apply_info:"""
        assert self.previous is not None
        assert (
            counter_apply_info.current == self.current
        ), f'{counter_apply_info.current} != {self.current}'
        return ApplyInfo(
            rule_name=self.rule_name,
            rule_formula=self._reverse_rule_formula(self.rule_formula),
            current=self.previous.current,
            # previous=dataclasses.replace(counter_apply_info, previous=None),
            previous=counter_apply_info,
            mapping=self.mapping,
            confidence=self.confidence,
            top=self.top,
            rule_id=self.rule_id,
            path=self.path,
        )

    def full_reverse(self, initial: Optional[ApplyInfo] = None) -> ApplyInfo:
        nodes = list(self.forward_trace)
        if len(nodes) < 2:
            return self
        nodes_copy = [node.clone() for node in nodes]
        for i in range(len(nodes) - 2, -1, -1):
            nodes[i] = dataclasses.replace(
                nodes_copy[i + 1],
                current=nodes[i].current,
                previous=nodes[i + 1],
                rule_formula=self._reverse_rule_formula(nodes_copy[i + 1].rule_formula),
            )
        nodes[-1] = initial or ApplyInfo.create_root(nodes[-1].current)
        nodes[-2].previous = nodes[-1]
        return nodes[0]

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

    def __str__(self):
        trace = [(s.current.verbose if s.current else 'None') for s in self.trace]
        return ' -> '.join(reversed(trace))

    @property
    def fit_info(self):
        if self.rule_id is None or self.path is None:
            return None
        return FitInfo(self.rule_id, self.path, self.contributed)

    @property
    def trace(self) -> Iterable[ApplyInfo]:
        step = self
        while step is not None:
            yield step
            step = step.previous

    @property
    def forward_trace(self) -> Iterable[ApplyInfo]:
        return reversed(list(self.trace))

    @property
    def track_loss(self):
        '''Number of steps to the last good value'''

        i = 0
        for i, step in enumerate(self.trace):
            if step.value is not None and step.value < 0.5:
                break

        return i


class ReversedLocalTrace:
    """Use this for bidirectional search."""

    def __init__(self, apply_infos: Iterable[ApplyInfo]) -> None:
        self._nodes = {apply_info.current: apply_info for apply_info in apply_infos}

    def __repr__(self) -> str:
        nodes = ', '.join(str(s) for s in self._nodes.keys())
        return f'ReversedLocalTrace{{{nodes}}}'

    def get_thread(self, apply_info: ApplyInfo) -> Optional[ApplyInfo]:
        # logging.debug(f'Looking for {apply_info.current} ...')
        towards_apply_info = self._nodes.get(apply_info.current, None)
        if towards_apply_info is None:
            return None

        return towards_apply_info.full_reverse(apply_info)


class LocalTrace:
    class Node:
        def __init__(self, apply_info: ApplyInfo):
            self.apply_info = apply_info
            self.childs = []

    def __init__(self, initial):
        self.root = LocalTrace.Node(ApplyInfo.create_root(initial))
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
        """Call this method before you search the next level."""
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

    def reversed(self) -> ReversedLocalTrace:
        return ReversedLocalTrace(self.iter())

    @staticmethod
    def as_dict_recursive(node):
        return {
            'apply_info': node.apply_info.as_dict(),
            'childs': [LocalTrace.as_dict_recursive(c) for c in node.childs],
        }

    def iter(self, just_neighborhood=False) -> Iterator[ApplyInfo]:
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
                if apply_info.rule_id is not None and apply_info.previous is not None:
                    # Just store fits of contributed steps
                    fits = (
                        [apply_info.fit_info]
                        if apply_info.previous.contributed
                        and apply_info.fit_info is not None
                        else []
                    )
                    if fits is not None:
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


def variable(ident: str) -> Symbol:
    return Symbol.variable(ident, True)


class TestApplyInfo(unittest.TestCase):
    def test_full_reverse(self):
        variables = [
            variable('a'),
            variable('b'),
            variable('c'),
            variable('d'),
            variable('e'),
        ]
        forward = ApplyInfo.create_root(variable('i'))
        for var in variables:
            forward = ApplyInfo.create_for_test(
                rule_name=f'rule to {var}',
                current=var,
                previous=forward,
            )
        # Check that the original thread as as expected
        self.assertEqual(str(forward), 'i -> a -> b -> c -> d -> e')
        rules = [a.rule_name for a in forward.forward_trace]
        self.assertEqual(
            rules,
            [
                'initial',
                'rule to a',
                'rule to b',
                'rule to c',
                'rule to d',
                'rule to e',
            ],
        )

        backward = forward.full_reverse()
        self.assertEqual(str(backward), 'e -> d -> c -> b -> a -> i')

        rules = [a.rule_name.replace('to', 'from') for a in backward.forward_trace]
        self.assertEqual(
            rules,
            [
                'initial',
                'rule from e',
                'rule from d',
                'rule from c',
                'rule from b',
                'rule from a',
            ],
        )
        # The original thread should not be modified.
        self.assertEqual(str(forward), 'i -> a -> b -> c -> d -> e')


class TestLocalTrace(unittest.TestCase):
    def test_trace(self):
        a = variable('a')
        b = variable('b')
        c = variable('c')
        local_trace = LocalTrace(a)

        apply_b = ApplyInfo.create_for_test(
            rule_name='first',
            current=b,
            previous=local_trace[0],
        )
        local_trace.add(apply_b)
        local_trace.close_stage()
        apply_c = ApplyInfo.create_for_test(
            rule_name='second',
            current=c,
            previous=local_trace[0],
        )

        local_trace.add(apply_c)
        trace = [s.current.verbose for s in local_trace.iter()]
        self.assertEqual(trace, ['a', 'b', 'c'])

    def test_reversed_short(self):
        a = variable('a')
        b = variable('b')
        c = variable('c')
        backward_trace = LocalTrace(c)
        apply_b = ApplyInfo.create_for_test(
            rule_name='reversed',
            current=b,
            previous=backward_trace[0],
        )

        backward_trace.add(apply_b)

        # c -> b   -->   b -> c
        reversed = backward_trace.reversed()

        forward_trace = LocalTrace(a)
        apply_b = ApplyInfo.create_for_test(
            rule_name='forward',
            current=b,
            previous=forward_trace[0],
        )

        thread = reversed.get_thread(apply_b)
        self.assertIsNotNone(thread)
        if thread is None:
            return

        self.assertEqual(str(thread), 'a -> b -> c')

    # @unittest.skip('Debugging')
    def test_reversed_long(self):
        a = variable('a')
        b = variable('b')
        c = variable('c')
        d = variable('d')
        e = variable('e')
        backward_trace = LocalTrace(e)
        apply_d = ApplyInfo.create_for_test(
            rule_name='first reversed',
            current=d,
            previous=backward_trace[0],
        )
        backward_trace.add(apply_d)
        backward_trace.close_stage()
        apply_c = ApplyInfo.create_for_test(
            rule_name='second reversed',
            current=c,
            previous=backward_trace[0],
        )
        backward_trace.add(apply_c)

        # e -> d -> c   -->   c -> d -> e
        reversed = backward_trace.reversed()

        forward_trace = LocalTrace(a)
        apply_b = ApplyInfo.create_for_test(
            rule_name='first forward',
            current=b,
            previous=forward_trace[0],
        )
        forward_trace.add(apply_b)
        forward_trace.close_stage()
        apply_c = ApplyInfo.create_for_test(
            rule_name='second forward',
            current=c,
            previous=forward_trace[0],
        )

        thread = reversed.get_thread(apply_c)
        self.assertIsNotNone(thread)
        if thread is None:
            return

        self.assertEqual(str(thread), 'a -> b -> c -> d -> e')
