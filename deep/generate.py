from itertools import permutations, islice
from string import ascii_lowercase as alphabet
from random import choices, shuffle, randint

from typing import List, Set, Dict, Tuple, Optional

from node import Node

from copy import deepcopy
from collections import deque


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

    def _leaves(self, level: int):
        nodes = [self]
        for _ in range(0, level):
            nodes = [child for node in nodes for child in node.childs]
        return nodes

    def _node_at(self, path: List[int]):
        node = self.childs[0]
        for dir in path:
            node = node.childs[dir]
        return node

    def set_idents_bfs(self, idents: List):
        queue = deque(self.childs)
        idents_index = 0
        while len(queue) > 0:
            node = queue.popleft()
            node.ident = idents[idents_index]
            idents_index += 1
            queue += node.childs

    def set_node(self, node: Node, path: List[int]):
        c_node = self.childs[0]
        for i in path[:-1]:
            c_node = c_node.childs[i]
        c_node[path[-1]] = node

    def set_idents_bfs_at(self, idents: List, path: List[int]):
        queue = deque([self._node_at(path)])
        for ident in idents:
            if len(queue) == 0:
                raise 'Not leaf not big enough'
            node = queue.popleft()
            node.ident = ident
            queue += node.childs

    def traverse_bfs_at(self, path: List[int]):
        queue = deque([self._node_at(path)])
        while len(queue) > 0:
            node = queue.popleft()
            queue += node.childs
            yield node

    def traverse_bfs(self):
        queue = deque([self.childs[0]])
        while len(queue) > 0:
            node = queue.popleft()
            queue += node.childs
            yield node

    def traverse_bfs_path(self):
        queue = deque([([], self.childs[0])])
        while len(queue) > 0:
            path, node = queue.popleft()
            queue += [(path+[i], n) for i, n in enumerate(node.childs)]
            yield path, node

    def add_level_uniform(self, child_per_arm: int):
        for leave in self._leaves(self.depth):
            leave.childs = [Node() for _ in range(0, child_per_arm)]
        self.depth += 1

    def has_pattern(self, pattern: List):
        '''Assuming homogenous spread'''
        if len(pattern) == 0:
            raise 'Pattern of length 0 is not supported'
        for path, node in self.traverse_bfs_path():
            if node.ident == pattern[0]:
                node_pattern = [
                    node.ident for node in self.traverse_bfs_at(path)]
                if list(islice(node_pattern, len(pattern))) == pattern:
                    return True
        return False

    def bfs_path(self, index: int):
        return next(islice(self.traverse_bfs_path(), index, index + 1))[0]

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


def _count_pattern(sequence, pattern):
    findings = 0
    plen = len(pattern)
    for i in range(len(sequence) - plen+1):
        if sequence[i:i+plen] == pattern:
            findings += 1
    return findings


def create_complex_pattern_in_noise(depth=4, spread=2, max_size=120, pattern_depth=1):
    '''Embeds a unique and fixed pattern (beginning of the alphabet) into noise'''

    if pattern_depth >= depth:
        raise f'Pattern depth ({pattern_depth}) must be smaller than outer depth ({depth})'
    samples = []

    builder = SymbolBuilder()
    for _ in range(depth-1):
        builder.add_level_uniform(spread)

    # Create pattern
    idents_reservoir = generate_idents()
    pattern_size = sum([spread**l for l in range(0, pattern_depth)])
    pattern = list(islice(idents_reservoir, pattern_size))
    # pattern_builder = SymbolBuilder()
    # for _ in range(pattern_depth):
    #     builder.add_level_uniform(spread)
    # pattern_size = sum([spread**l for l in range(0, pattern_depth+1)])
    # idents = list(islice(idents_reservoir, pattern_size))
    # pattern_builder.set_idents_bfs(idents)
    # pattern = pattern_builder.symbol

    # Noise
    idents_reservoir = generate_idents()

    size = sum([spread**l for l in range(0, depth)])
    idents = list(islice(idents_reservoir, size))
    classes = []

    max_depth = depth - pattern_depth+1
    pos_count = pattern_size = sum(
        [spread**l for l in range(0, max_depth)])

    classes_dict = {}

    while len(samples) < max_size:
        shuffled = deepcopy(idents)
        shuffle(shuffled)

        builder.set_idents_bfs(shuffled)
        if builder.has_pattern(pattern):
            continue
        # Find position
        pos = randint(0, pos_count-1)
        path = builder.bfs_path(pos)
        builder.set_idents_bfs_at(pattern, path)

        if pos not in classes_dict:
            classes_dict[pos] = len(classes_dict)
            classes.append(classes_dict[pos])
        class_id = classes_dict[pos]

        samples.append((class_id, builder.symbol))

    return samples, idents, classes
