from itertools import permutations, islice
from string import ascii_lowercase as alphabet
from random import choices, shuffle

from typing import List, Set, Dict, Tuple, Optional

from node import Node

from copy import deepcopy


def generate_idents():
    # First use one char of alpahabet
    for c in alphabet:
        yield c

    for c in alphabet:
        for cc in alphabet:
            yield c + cc


class SymbolBuilder:
    def __init__(self):
        self.childs = [Node()]
        self.depth = 1

    def _leaves(self, level):
        nodes = [self]
        for _ in range(0, level):
            nodes = [child for node in nodes for child in node.childs]
        return nodes

    def set_level_idents(self, level, idents):
        for (ident, leave) in zip(idents, self._leaves(self.depth)):
            leave.ident = ident

    def set_idents_bfs(self, idents):
        queue = []
        queue += self.childs
        idents_index = 0
        while len(queue) > 0:
            node = queue.pop()
            node.ident = idents[idents_index]
            idents_index += 1
            queue += node.childs

    def set_node(self, node: Node, path: List[int]):
        c_node = self.childs[0]
        for i in path[:-1]:
            c_node = c_node.childs[i]
        c_node[path[-1]] = node

    def add_level_uniform(self, child_per_arm):
        for leave in self._leaves(self.depth):
            leave.childs = [Node() for _ in range(0, child_per_arm)]
        self.depth += 1

    @property
    def symbol(self):
        return deepcopy(self.childs[0])


def create_samples_permutation(depth=2, spread=1, max_size=120):
    '''
    Samples are of the form (class_id, symbol)
    Using a permutation where of unique idents
    '''
    samples = []

    size = sum([spread**l for l in range(0, depth+1)])

    builder = SymbolBuilder()
    for _ in range(depth):
        builder.add_level_uniform(spread)

    idents = list(islice(generate_idents(), size))
    classes = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)

        builder.set_idents_bfs(perm)

        samples.append((i, builder.symbol))

        if len(samples) >= max_size:
            break

    return samples, idents, classes
