from typing import List, Set, Dict, Tuple, Optional
from itertools import islice
from copy import deepcopy
from collections import deque

from deep.node import Node


class SymbolBuilder:
    def __init__(self, root=None):
        self.childs = [root or Node()]
        self.depth = 0

    def _leaves(self, level: int):
        nodes = [self]
        for _ in range(0, level+1):
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
            yield node
            queue += node.childs

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
            yield path, node
            queue += [(path+[i], n) for i, n in enumerate(node.childs)]

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
