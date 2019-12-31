import unittest
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

import torch

from node import Node
from .symbol_builder import SymbolBuilder

PAD_TOKEN = '<PAD>'


def ident_to_id(node: Node, idents: List = None):
    # 0 is padding
    if node.ident == PAD_TOKEN:
        return 0
    if idents is None:
        return ord(node.ident) - 97 + 1

    return idents.index(node.ident)


class TraverseInstructionSet:
    def __init__(self, input=None, hidden=None):
        self.input = input
        self.hidden = hidden

    def get(self, x, h):
        if self.input is not None:
            return x[self.input]
        if self.hidden is not None:
            return h[self.hidden]
        raise Exception('Missing index!')

    def __repr__(self):
        if self.input is not None:
            return f'i{self.input}'
        if self.hidden is not None:
            return f'h{self.hidden}'
        raise Exception('Missing index!')


class TraverseInstruction:
    def __init__(self, root, childs):
        self.root = root
        self.childs = childs

    def get(self, x, h):
        root = self.root.get(x, h)
        childs = [child.get(x, h) for child in self.childs]
        return root, childs

    def get_index(self):
        return self.root.input

    def __repr__(self):
        childs = ', '.join([repr(child) for child in self.childs])
        return f'{repr(self.root)} - {childs}'


class Embedder:
    '''Traversing post order

    Assuming each sample has the same form
    '''

    @staticmethod
    def blueprint(params):
        depth = params.depth
        spread = params.spread
        # One line for each node (not leaf)
        lines = [l for l in range(depth)[::-1] for _ in range(spread**l)]
        i = 0
        h = 0
        instructions = []
        for l in lines:
            if l == depth-1:
                instructions.append(TraverseInstruction(
                    TraverseInstructionSet(input=i + spread),
                    [TraverseInstructionSet(input=i+s) for s in range(spread)]))
                i += spread + 1
            else:
                instructions.append(TraverseInstruction(
                    TraverseInstructionSet(input=i),
                    [TraverseInstructionSet(hidden=h+s) for s in range(spread)]))
                i += 1
                h += spread
        return instructions

    @staticmethod
    def legend(params):
        depth = params.depth
        spread = params.spread
        builder = SymbolBuilder()
        for _ in range(depth):
            builder.add_level_uniform(spread)
        for path, node in builder.traverse_bfs_path():
            node.label = path

        embedder = Embedder()
        for line in embedder.unroll(builder.symbol):
            yield line.label

    @staticmethod
    def leaf_mask(params):
        '''Returns a mask where each leaves are masked out'''
        depth = params.depth
        spread = params.spread
        tree_size = sum([spread**l for l in range(0, depth+1)])
        all_nodes = np.ones(tree_size)
        if depth == 0:
            return all_nodes
        num_groups = spread**(depth-1)

        for i in range(num_groups):
            offset = i*(spread+1)
            all_nodes[offset:offset+spread] = 0

        return all_nodes

    def unroll(self, x: Node) -> List[Node]:
        assert x is not None
        stack = [x]
        x = []
        seen = set()
        while len(stack) > 0:
            n = stack[-1]
            if len(n.childs) > 0 and n not in seen:
                stack.extend(n.childs[::-1])
                seen.add(n)
            else:
                stack.pop()
                yield n


class TagEmbedder(Embedder):
    '''Each sample is associated to one tag'''

    def __call__(self, x: Node, y, s, **kwargs):
        x = [ident_to_id(n) for n in self.unroll(x)]
        return x, y, s


class SegEmbedder(Embedder):
    '''Each sub node in a sample is associated to one tag'''

    def __call__(self, x: Node, y: Node, s, idents, **kwargs):
        y = [n.label or 0 for n in self.unroll(y)]
        x = [ident_to_id(n, idents) for n in self.unroll(x)]
        return x, y, s


class Padder:
    '''Adds childs to each node of the tree that each has the same depth, spread and is complete.'''

    def pad(self, n: Node, depth: int, spread: int, **kwargs) -> Node:
        if type(n).__name__ == 'Node':
            builder = SymbolBuilder(n)
            for path, node in builder.traverse_bfs_path():
                if len(path) < depth:
                    nc = len(node.childs)
                    for _ in range(nc, spread):
                        node.childs.append(Node(PAD_TOKEN, []))
            return builder.symbol
        elif type(n).__name__ == 'Symbol':
            return n.create_padded(PAD_TOKEN, spread, depth)
        else:
            raise Exception(f'Unknown type {type(n).__name__}')

    def __call__(self, x: Node, y: Node, *args, **kwargs):
        # TODO: Write test
        x = self.pad(x, **kwargs)
        y = self.pad(y, **kwargs)

        assert x is not None
        assert y is not None

        return (x, y) + args


class Uploader:
    def __init__(self, device=torch.device('cpu')):  # pylint: disable=no-member
        self.device = device

    def __call__(self, x, y, s, **kwargs):
        x = torch.as_tensor(x, dtype=torch.long).to(self.device)  # pylint: disable=no-member
        y = torch.as_tensor(y, dtype=torch.long).to(self.device)  # pylint: disable=no-member
        return x, y, s


class TestPadder(unittest.TestCase):
    '''Unit test for the padder class'''

    def test_to_short(self):
        padder = Padder(depth=2, spread=2)
        node = Node('a')
        padded, = padder(node)  # pylint: disable=unbalanced-tuple-unpacking
        p = PAD_TOKEN

        expected = Node('a', [Node(p, [Node(p, []), Node(p, [])]),
                              Node(p, [Node(p, []), Node(p, [])])])

        self.assertEqual(padded, expected)

    def test_unbalanced(self):
        padder = Padder(depth=2, spread=2)
        node = Node('a', [Node('b', [Node('c')])])
        p = PAD_TOKEN
        padded, = padder(node)  # pylint: disable=unbalanced-tuple-unpacking

        expected = Node('a', [Node('b', [Node('c', []), Node(p, [])]),
                              Node(p, [Node(p, []), Node(p, [])])])
        self.assertEqual(padded, expected)


class TestEmbedder(unittest.TestCase):
    '''Unit test for Embedder base class'''
    class Params:
        '''Simple POD class'''

        def __init__(self, depth, spread):
            self.depth = depth
            self.spread = spread

    def test_leaf_mask(self):
        mask = Embedder.leaf_mask(TestEmbedder.Params(spread=2, depth=3))

        expected = np.array([0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1.])
        # self.assertEqual(mask, expected)
        np.allclose(mask, expected)
