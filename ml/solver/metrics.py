from typing import List
import logging

from solver.trace import ApplyInfo

logger = logging.getLogger(__name__)


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

    def as_dict(self):
        return {"values": self.values, "total": self.total, "worst": self.worst}

    @property
    def worst(self):
        if len(self.values) > 0:
            return max(self.values.keys())
        return 0


def calculate_policy_tops(solutions: List[ApplyInfo], tops: Tops = None):
    tops = tops or Tops()
    for solution in solutions:
        for step in solution.trace:
            tops += step.top

    return tops


def calculate_value_tops(solutions: List[ApplyInfo], tops: Tops = None):
    tops = tops or Tops()

    for solution in solutions:
        for step in solution.trace:
            if step.previous is None:
                continue
            if step.value is None:
                continue

            sisters = [s for s in step.previous.subsequent if s.value]
            sisters.sort(key=lambda info: info.value, reverse=True)
            try:
                self_index = next(i for i, sister in enumerate(sisters) if sister is step)
            except StopIteration:
                msg = f'Could not find myself {step.current.verbose} as a children of my parent: {[c.current.verbose for c in step.previous.subsequent]}'
                logger.error(msg)
                continue
                # raise RuntimeError(msg)
            tops += self_index

    return tops


def solution_summary(solutions: List[ApplyInfo], as_dict=False):
    if as_dict:
        return {
            'policy': calculate_policy_tops(solutions).as_dict(),
            'value': calculate_value_tops(solutions).as_dict()
        }
    return {
        'policy': calculate_policy_tops(solutions),
        'value': calculate_value_tops(solutions)
    }


class SolutionSummarieser:
    '''The same as solution function but accumulative'''

    def __init__(self):
        self.policy_tops = Tops()
        self.value_tops = Tops()

    def __add__(self, solution: ApplyInfo):
        self.policy_tops = calculate_policy_tops([solution], self.policy_tops)
        self.value_tops = calculate_value_tops([solution], self.value_tops)
        return self

    def summary(self, as_dict=False):
        if as_dict:
            return {
                'policy': self.policy_tops.as_dict(),
                'value': self.value_tops.as_dict()
            }
        return {
            'policy': self.policy_tops,
            'value': self.value_tops
        }
