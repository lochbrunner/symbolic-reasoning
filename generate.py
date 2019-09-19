from itertools import permutations
from string import ascii_lowercase as alphabet
from random import choices, shuffle

from copy import deepcopy
import json


class NodeEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Node:
    def __init__(self):
        self.childs = []
        self.ident = None

    def __repr__(self):
        return json.dumps(self, cls=NodeEncoder)


class SymbolBuilder:
    def __init__(self):
        self.childs = []
        self.depth = 0

    def _leaves(self, level):
        nodes = [self]
        for _ in range(0, level):
            nodes = [child for node in nodes for child in node.childs]
        return nodes

    def set_level_idents(self, level, idents):
        for (ident, leave) in zip(idents, self._leaves(self.depth)):
            leave.ident = ident

    def add_level_uniform(self, child_per_arm):
        for leave in self._leaves(self.depth):
            leave.childs = [Node() for _ in range(0, child_per_arm)]
        self.depth += 1

    @property
    def symbol(self):
        return deepcopy(self.childs)


def create_samples(level=1, spread=2, noise=2):
    '''Samples are of the form (class_id, symbol)'''
    samples = []

    size = spread**level

    builder = SymbolBuilder()
    for _ in range(level):
        builder.add_level_uniform(spread)

    idents = alphabet[:size]
    classes = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)
        builder.set_level_idents(level, perm)

        # Add noise to the upper nodes
        for _ in range(noise):
            for l in range(level):
                level_size = spread**l
                level_idents = choices(idents, k=level_size)
                builder.set_level_idents(l, level_idents)

            samples.append((i, builder.symbol))

    return samples, list(idents), classes
