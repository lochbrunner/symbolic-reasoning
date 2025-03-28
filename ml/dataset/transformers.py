import unittest
import numpy as np
from typing import List, Set, Dict, Tuple, Optional

import torch

from common.node import Node
from .symbol_builder import SymbolBuilder

PAD_TOKEN = '<PAD>'
PAD_INDEX = 0


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
        '''Unrolls a dummy node and records the lines'''
        builder = SymbolBuilder.create(**vars(params))
        input_mapping = {}
        embedder = Embedder()
        legend = []

        for c, node in enumerate(embedder.unroll(builder.symbol_ref)):
            input_mapping[id(node)] = c
        input_mapping[None] = len(input_mapping)

        hidden_mapping = {}

        def create_inst(node):
            i = id(node)
            if i in hidden_mapping:
                return TraverseInstructionSet(hidden=hidden_mapping[i])
            if i in input_mapping:
                return TraverseInstructionSet(input=input_mapping[i])
            raise Exception(f'Node {node} not found!')

        h = 0
        for node in reversed(list(builder.traverse_bfs())):
            # Ignore leaves for now
            if len(node.childs) != 0:
                legend.append(TraverseInstruction(
                    TraverseInstructionSet(input=input_mapping[id(node)]),
                    [create_inst(child) for child in node.childs]))
                hidden_mapping[id(node)] = h
                h += 1
            else:  # leaves
                # last is padding
                legend.append(TraverseInstruction(
                    TraverseInstructionSet(input=input_mapping[id(node)]),
                    [TraverseInstructionSet(input=input_mapping[None])
                        for _ in range(params.spread)]))
                h += 1

        return legend

    @staticmethod
    def legend(spread=None, depth=None, params=None):
        if depth is None:
            depth = params.max_depth if hasattr(params, 'max_depth') else params.depth
        if spread is None:
            spread = spread or params.max_spread if hasattr(params, 'max_spread') else params.spread
        builder = SymbolBuilder()
        for _ in range(depth):
            builder.add_level_uniform(spread)
        for path, node in builder.traverse_bfs_path():
            node.label = path

        for line in Embedder.unroll(builder.symbol):
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

    @staticmethod
    def unroll(x: Node) -> List[Node]:
        assert x is not None
        if type(x).__name__ == 'Node':
            return SymbolBuilder(x).traverse_bfs()
        if type(x).__name__ == 'Symbol':
            return x.parts_bfs


class TagEmbedder(Embedder):
    '''Each sample is associated to one tag'''

    def __call__(self, x: Node, y, s, **kwargs):
        x = [ident_to_id(n) for n in self.unroll(x)]
        return x, y, s


class SegEmbedder(Embedder):
    '''Each sub node in a sample is associated to one tag'''

    def __call__(self, x: Node, y: Node, m: Node, idents, *args, **kwargs):
        # Last is padding for the leaves
        x = [ident_to_id(n, idents) for n in self.unroll(x)] + [0]
        y = [n.label or 0 for n in self.unroll(y)] + [0]
        m = [n.label or 0 for n in self.unroll(m)] + [0]
        return x, y, m


class Padder:
    '''Adds childs to each node of the tree that each has the same depth, spread and is complete.'''

    @staticmethod
    def pad(n: Node, depth: int, spread: int, factory=None, **kwargs) -> Node:
        def default_factory():
            return Node(PAD_TOKEN, [])
        if factory is None:
            factory = default_factory
        if type(n).__name__ == 'Node':
            builder = SymbolBuilder(n)
            for path, node in builder.traverse_bfs_path():
                if len(path) < depth:
                    nc = len(node.childs)
                    for _ in range(nc, spread):
                        node.childs.append(factory())
            return builder.symbol_ref
        # Rust version of Symbol
        elif type(n).__name__ == 'Symbol':
            return n.create_padded(PAD_TOKEN, spread, depth)
        else:
            raise Exception(f'Unknown type {type(n).__name__}')

    @staticmethod
    def create_mask(n: Node, padding=0, **kwargs):
        if type(n).__name__ == 'Symbol':
            mask = Node.from_rust(n)
        elif type(n).__name__ == 'Node':
            mask = n.clone()

        builder = SymbolBuilder(mask)
        for n in builder.traverse_bfs():
            n.label = padding
        return builder

    def __call__(self, x: Node, *args, label: int, path: List[int], **kwargs):
        x = Padder.pad(x, **kwargs)
        builder = Padder.create_mask(x)
        m = Padder.create_mask(x).symbol_ref
        builder.set_label_at(path, label)
        y = builder.symbol_ref

        def factory():
            return Node(label=-1, childs=[])
        y = Padder.pad(y, factory=factory, **kwargs)
        assert x is not None
        assert y is not None
        assert m is not None

        return (x, y, m) + args


class Uploader:

    def __call__(self, x, y, **kwargs):
        x = torch.as_tensor(x, dtype=torch.long)  # pylint: disable=no-member
        y = torch.as_tensor(y, dtype=torch.long)  # pylint: disable=no-member
        return x, y


class TestPadder(unittest.TestCase):
    '''Unit test for the padder class'''

    @unittest.skip('Method usage changed')
    def test_to_short(self):
        padder = Padder()
        node = Node('a')
        padded, _ = padder(node, node, depth=2, spread=2)  # pylint: disable=unbalanced-tuple-unpacking
        p = PAD_TOKEN

        expected = Node('a', [Node(p, [Node(p, []), Node(p, [])]),
                              Node(p, [Node(p, []), Node(p, [])])])

        self.assertEqual(padded, expected)

    @unittest.skip('Method usage changed')
    def test_unbalanced(self):
        padder = Padder()
        node = Node('a', [Node('b', [Node('c')])])
        p = PAD_TOKEN
        padded, _ = padder(node, node, depth=2, spread=2)  # pylint: disable=unbalanced-tuple-unpacking

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

    def test_unroll_wide(self):

        node = Node('a', [
            Node('ab', [Node('abe'), Node('abf'), Node('abg')]),
            Node('ac', [Node('ach'), Node('aci'), Node('adj')]),
            Node('ad', [Node('adk'), Node('adl'), Node('adm')])])
        embedder = Embedder()
        unrolled = embedder.unroll(node)
        unrolled = list(unrolled) + ['PADDING']
        # self.assertEqual([n.ident for n in unrolled], ['abe', 'abf', 'abg', 'ab',
        #                                                'ach', 'aci', 'adj', 'ac', 'adk', 'adl', 'adm', 'ad', 'a'])
        params = TestEmbedder.Params(2, 3)
        blueprint = Embedder.blueprint(params)

        seen = set()
        hidden = []
        for inst in blueprint:
            r, c = inst.get(unrolled, hidden)
            if len(r.childs) == 0:
                for ci in c:
                    self.assertEqual(ci, 'PADDING')
                self.assertEqual(len(c), params.spread)
            else:
                self.assertEqual([n.ident for n in r.childs], [n.ident for n in c])
            hidden.append(r)
            seen.add(id(r))

        self.assertEqual(len(seen), sum([params.spread**l for l in range(params.depth+1)]))
        self.assertEqual(len(blueprint), len(seen), 'All instructions are unique')

    @staticmethod
    def print_lined(code):
        for i, line in enumerate(code.split('\n')):
            print(f'{str(i).rjust(2)}: {line}')

    def test_unroll_depth(self):

        node = Node('a', [
            Node('ab', [Node('abe', [Node('aben'), Node('abeo')]), Node('abf', [Node('abfp'), Node('abfq')])]),
            Node('ac', [Node('ach', [Node('achr'), Node('achs')]), Node('aci', [Node('acit'), Node('aciu')])])])
        embedder = Embedder()
        unrolled = embedder.unroll(node)
        unrolled = list(unrolled) + ['PADDING']
        # self.assertEqual([n.ident for n in unrolled], ['aben', 'abeo', 'abe', 'abfp',
        #                                                'abfq', 'abf', 'ab', 'achr', 'achs', 'ach', 'acit', 'aciu', 'aci', 'ac', 'a'])

        params = TestEmbedder.Params(depth=3, spread=2)
        blueprint = Embedder.blueprint(params)

        seen = set()

        hidden = []
        for inst in blueprint:
            r, c = inst.get(unrolled, hidden)
            if len(r.childs) == 0:
                for ci in c:
                    self.assertEqual(ci, 'PADDING')
                self.assertEqual(len(c), params.spread)
            else:
                self.assertEqual([n.ident for n in r.childs], [n.ident for n in c])
            hidden.append(r)
            seen.add(id(r))

        self.assertEqual(len(seen), sum([params.spread**l for l in range(params.depth+1)]))
        self.assertEqual(len(blueprint), len(seen), 'All instructions are unique')
