from typing import List, Set, Dict, Tuple, Optional
from itertools import islice
from copy import deepcopy
from collections import deque
import unittest

from node import Node


class SymbolBuilder:
    def __init__(self, root=None):
        self.childs = [root or Node()]
        self.depth = 0

    def _leaves(self, level: int):
        nodes = [self]
        for _ in range(0, level+1):
            nodes = [child for node in nodes for child in node.childs]
        return nodes

    def node_at(self, path: List[int]):
        node = self.childs[0]
        for d in path:
            node = node.childs[d]
        return node

    def _traverse_bfs(self, begin):
        queue = deque([begin])
        while len(queue) > 0:
            node = queue.popleft()
            yield node
            queue += node.childs

    def traverse_bfs(self):
        return self._traverse_bfs(self.childs[0])

    def traverse_bfs_at(self, path: List[int]):
        return self._traverse_bfs(self.node_at(path))

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
            yield path, node
            queue += [(path+[i], n) for i, n in enumerate(node.childs)]

    def add_level_uniform(self, child_per_arm: int):
        for leave in self._leaves(self.depth):
            leave.childs = [Node(parent=leave) for _ in range(0, child_per_arm)]
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
        self.node_at(path).label = label

    @property
    def symbol(self):
        return deepcopy(self.childs[0])

    @property
    def symbol_ref(self):
        return self.childs[0]

    @staticmethod
    def create(depth, spread, **kwargs):
        builder = SymbolBuilder()
        for _ in range(depth):
            builder.add_level_uniform(spread)
        return builder


class TestStringBuilder(unittest.TestCase):
    def test_traverse_bfs(self):
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        actual = list([node.ident for node in builder.traverse_bfs()])
        expected = ['a', 'b', 'e', 'c', 'd']

        self.assertEqual(actual, expected)

    def test_traverse_bfs_path(self):
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder = SymbolBuilder(node)

        actual = list([path for path, node in builder.traverse_bfs_path()])
        expected = [[], [0], [1], [0, 0], [0, 1]]
        self.assertEqual(actual, expected)

    def test_set_idents_bfs(self):
        builder = SymbolBuilder()
        node = Node('n', [Node('n', [Node('n'), Node('n')]), Node('n')])
        builder.childs = [node]
        builder.set_idents_bfs(['a', 'b', 'c', 'd', 'e'])

        expected = Node('a', [Node('b', [Node('d'), Node('e')]), Node('c')])

        self.assertEqual(builder.symbol, expected)

    def test_traverse_bfs_at(self):
        # Using tree:
        # a
        #  b
        #   c
        #   d
        #  e
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        actual = list([node.ident for node in builder.traverse_bfs_at([0])])
        expected = ['b', 'c', 'd']

        self.assertEqual(actual, expected)

    def test_set_idents_bfs_at(self):
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        builder.set_idents_bfs_at(['r', 's', 't'], [0])

        actual = list([node.ident for node in builder.traverse_bfs()])
        expected = ['a', 'r', 'e', 's', 't']

        self.assertEqual(actual, expected)

    def test_has_pattern_negative(self):
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        pattern = ['b', 'd', 'c']
        self.assertFalse(builder.has_pattern(pattern))

    def test_has_pattern_positive(self):
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        pattern = ['b', 'c', 'd']
        self.assertTrue(builder.has_pattern(pattern))

    def test_bfs_path(self):
        builder = SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        actual = builder.bfs_path(4)
        expected = [0, 1]
        self.assertEqual(actual, expected)
