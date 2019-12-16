from itertools import permutations, islice
from string import ascii_lowercase as alphabet
from random import choices, choice, shuffle, randint

from typing import List, Set, Dict, Tuple, Optional

from deep.node import Node

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
    def __init__(self, root=Node()):
        self.childs = [root]
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

    def _traverse_bfs(self, begin):
        queue = deque([begin])
        while len(queue) > 0:
            node = queue.popleft()
            queue += node.childs
            yield node

    def traverse_bfs(self):
        return self._traverse_bfs(self.childs[0])

    def traverse_bfs_at(self, path: List[int]):
        return self._traverse_bfs(self._node_at(path))

    def set_node(self, node: Node, path: List[int]):
        c_node = self.childs[0]
        for i in path[:-1]:
            c_node = c_node.childs[i]
        c_node[path[-1]] = node

    def set_idents_bfs(self, idents: List):
        for ident, node in zip(idents, self.traverse_bfs()):
            node.ident = ident

    def set_idents_bfs_at(self, idents: List, path: List[int]):
        for ident, node in zip(idents, self.traverse_bfs_at(path)):
            node.ident = ident

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

    def find_pattern(self, pattern: List):
        '''Assuming homogenous spread'''
        if len(pattern) == 0:
            raise 'Pattern of length 0 is not supported'
        for path, node in self.traverse_bfs_path():
            if node.ident == pattern[0]:
                node_pattern = [
                    node.ident for node in self.traverse_bfs_at(path)]
                if list(islice(node_pattern, len(pattern))) == pattern:
                    return path
        return None

    def has_pattern(self, pattern: List):
        '''Assuming homogenous spread'''
        return self.find_pattern(pattern) is not None

    def bfs_path(self, index: int):
        return next(islice(self.traverse_bfs_path(), index, index + 1))[0]

    def clear_labels(self):
        for node in self.traverse_bfs():
            node.label = None

    def set_label_at(self, path: List[int], label):
        self._node_at(path).label = label

    @property
    def symbol(self):
        return deepcopy(self.childs[0])


def scenarios_choices():
    return ['permutation', 'pattern']


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
    # This line is needed because of python bug while running the unit tests.
    builder.clear_labels()

    idents = list(islice(generate_idents(), size))
    classes = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)

        builder.set_idents_bfs(perm)

        samples.append((i, builder.symbol))

        if len(samples) >= max_size:
            break

    return samples, idents, classes


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


def place_patterns_in_noise(depth=4, spread=2, max_size=120, pattern_depth=1, num_labels=5):
    '''Embeds some unique and fixed patterns (beginning of the alphabet) into noise'''

    if pattern_depth >= depth:
        raise f'Pattern depth ({pattern_depth}) must be smaller than outer depth ({depth})'
    samples = []

    builder = SymbolBuilder()
    for _ in range(depth-1):
        builder.add_level_uniform(spread)

    # Noise
    idents_reservoir = generate_idents()

    tree_size = sum([spread**l for l in range(0, depth)])
    idents = list(islice(idents_reservoir, tree_size))

    # Create pattern
    idents_reservoir = generate_idents()
    pattern_size = sum([spread**l for l in range(0, pattern_depth)])
    if pattern_size*num_labels > tree_size:
        raise Exception(f'Not enough space to place {num_labels} different patterns!')
    patterns = [list(islice(idents_reservoir, pattern_size)) for _ in range(num_labels)]

    max_depth = depth - pattern_depth+1
    pos_count = pattern_size = sum(
        [spread**l for l in range(0, max_depth)])

    max_tries = max_size*10
    while len(samples) < max_size:
        shuffled = deepcopy(idents)
        shuffle(shuffled)

        builder.set_idents_bfs(shuffled)
        for pattern in patterns:
            if builder.has_pattern(pattern):
                continue

        # Find position
        label_id = randint(0, len(patterns)-1)
        pattern = patterns[label_id]

        pos = randint(0, pos_count-1)
        path = builder.bfs_path(pos)
        builder.set_idents_bfs_at(pattern, path)
        builder.clear_labels()
        # Label 0 means no label
        builder.set_label_at(path, label_id+1)

        samples.append(builder.symbol)

        max_tries -= 1
        if max_tries == 0:
            raise Exception(f'Could not find enough samples.')

    return samples, idents, patterns
