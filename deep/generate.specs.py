#!/usr/bin/env python3
from node import Node
import generate
import unittest


def calc_depth(node: Node):
    depth = 0
    stack = [node]
    while len(stack) > 0:
        node = stack.pop()
        depth += 1
        if len(node.childs) > 0:
            stack += node.childs[0:1]
        else:
            break

    return depth


# @unittest.skip('Debugging')
class TestPermutation(unittest.TestCase):
    def test_flat(self):
        samples, idents, classes = generate.create_samples_permutation(
            depth=1, spread=1)

        self.assertEqual(idents, ['a', 'b'])
        self.assertEqual(classes, [0, 1])

        self.assertEqual(len(samples), 2)

        expected = Node()
        expected.ident = 'a'
        expected.childs = [Node('b')]
        self.assertEqual(expected, samples[0][1])

        expected = Node()
        expected.ident = 'b'
        expected.childs = [Node('a')]
        self.assertEqual(expected, samples[1][1])

    def test_wide(self):
        samples, idents, classes = generate.create_samples_permutation(
            depth=1, spread=2)

        self.assertEqual(len(samples), 6)

        expected_samples = []
        expected_samples.append(Node('a', [Node('b'), Node('c')]))
        expected_samples.append(Node('a', [Node('c'), Node('b')]))
        expected_samples.append(Node('b', [Node('a'), Node('c')]))
        expected_samples.append(Node('b', [Node('c'), Node('a')]))
        expected_samples.append(Node('c', [Node('a'), Node('b')]))
        expected_samples.append(Node('c', [Node('b'), Node('a')]))

        self.maxDiff = None
        self.assertCountEqual([sample[1]
                               for sample in samples], expected_samples)

        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertEqual(classes, [0, 1, 2, 3, 4, 5])

    def test_deep(self):
        samples, idents, classes = generate.create_samples_permutation(
            depth=2, spread=1)

        self.assertEqual(len(samples), 6)

        expected_samples = []
        expected_samples.append(Node('a', [Node('b', [Node('c')])]))
        expected_samples.append(Node('a', [Node('c', [Node('b')])]))
        expected_samples.append(Node('b', [Node('a', [Node('c')])]))
        expected_samples.append(Node('b', [Node('c', [Node('a')])]))
        expected_samples.append(Node('c', [Node('a', [Node('b')])]))
        expected_samples.append(Node('c', [Node('b', [Node('a')])]))

        self.maxDiff = None
        self.assertCountEqual([sample[1]
                               for sample in samples], expected_samples)

        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertEqual(classes, [0, 1, 2, 3, 4, 5])


class TestPattern(unittest.TestCase):
    def test_slim_pattern(self):
        samples, idents, classes = generate.create_complex_pattern_in_noise(
            depth=3, spread=1, max_size=2, pattern_depth=2)

        builder = generate.SymbolBuilder()

        builder.childs = [samples[0][1]]
        self.assertTrue(builder.has_pattern(['a', 'b']))

        builder.childs = [samples[1][1]]
        self.assertTrue(builder.has_pattern(['a', 'b']))

        self.assertEqual(calc_depth(samples[0][1]), 3)
        self.assertEqual(calc_depth(samples[1][1]), 3)
        self.assertEqual(len(samples), 2)
        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertGreaterEqual(len(classes), 1)

    def test_wide_pattern(self):
        samples, idents, classes = generate.create_complex_pattern_in_noise(
            depth=3, spread=2, max_size=2, pattern_depth=2)

        builder = generate.SymbolBuilder()

        builder.childs = [samples[0][1]]
        self.assertTrue(builder.has_pattern(['a', 'b', 'c']))

        builder.childs = [samples[1][1]]
        self.assertTrue(builder.has_pattern(['a', 'b', 'c']))

        self.assertEqual(calc_depth(samples[0][1]), 3)
        self.assertEqual(calc_depth(samples[1][1]), 3)
        self.assertEqual(len(samples), 2)
        self.assertEqual(idents, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        self.assertGreaterEqual(len(classes), 1)


# @unittest.skip('Debugging')
class TestStringBuilder(unittest.TestCase):
    def test_traverse_bfs(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        actual = list([node.ident for node in builder.traverse_bfs()])
        expected = ['a', 'b', 'e', 'c', 'd']

        self.assertEqual(actual, expected)

    def test_traverse_bfs_path(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        actual = list([path for path, node in builder.traverse_bfs_path()])
        expected = [[], [0], [1], [0, 0], [0, 1]]
        self.assertEqual(actual, expected)

    def test_traverse_bfs_at(self):
        # Using tree:
        # a
        #  b
        #   c
        #   d
        #  e
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        actual = list([node.ident for node in builder.traverse_bfs_at([0])])
        expected = ['b', 'c', 'd']

        self.assertEqual(actual, expected)

    def test_set_idents_bfs_at(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])

        builder.childs = [node]

        builder.set_idents_bfs_at(['r', 's', 't'], [0])

        actual = list([node.ident for node in builder.traverse_bfs()])
        expected = ['a', 'r', 'e', 's', 't']

        self.assertEqual(actual, expected)

    def test_has_pattern_negative(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        pattern = ['b', 'd', 'c']
        self.assertFalse(builder.has_pattern(pattern))

    def test_has_pattern_positive(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        pattern = ['b', 'c', 'd']
        self.assertTrue(builder.has_pattern(pattern))

    def test_bfs_path(self):
        builder = generate.SymbolBuilder()
        node = Node('a', [Node('b', [Node('c'), Node('d')]), Node('e')])
        builder.childs = [node]

        actual = builder.bfs_path(4)
        expected = [0, 1]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
