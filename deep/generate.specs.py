#!/usr/bin/env python3
import unittest

import generate
from node import Node


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


if __name__ == '__main__':
    unittest.main()
