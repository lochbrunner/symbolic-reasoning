#!/usr/bin/env python3

from pycore import Sample, SampleSet, Symbol, Context, FitInfo

import unittest


class TestSample(unittest.TestCase):
    def test_single_embed(self):
        context = Context.standard()
        initial = Symbol.parse(context, 'a/(x+1)=a-1')
        fits = [FitInfo(1, [0, 0], True), FitInfo(2, [0, 1], False)]
        sample = Sample(initial, fits, useful=True)

        ident2index = {'a': 1, '/': 2, 'x': 3, '+': 4, '1': 5, '=': 6, '-': 7}
        padding = 20
        _, _, _, label, policy, value = sample.embed(
            ident2index, padding, 2, initial.depth, index_map=True, positional_encoding=False)

        self.assertEqual(value.tolist(), [1])
        self.assertEqual(policy.tolist(), [0, 0, 0, 1, -1, 0, 0, 0, 0, 0])
        self.assertEqual(label.tolist(), [0, 0, 0, 1, 2, 0, 0, 0, 0, 0])

    def test_from_merged(self):
        context = Context.standard()
        initial = Symbol.parse(context, 'a/(x+1)=a-1')

        sample = Sample(
            initial=initial,
            fits=[
                FitInfo(rule_id=1, path=[0, 0], positive=True),
                FitInfo(rule_id=2, path=[0, 1], positive=False),
                FitInfo(rule_id=1, path=[1, 0], positive=False),
                FitInfo(rule_id=2, path=[1, 1], positive=True),
            ],
            useful=True
        )

        ident2index = {'a': 1, '/': 2, 'x': 3, '+': 4, '1': 5, '=': 6, '-': 7}
        padding = 20
        _, _, _, label, policy, value = sample.embed(
            ident2index, padding, 2, initial.depth, index_map=True, positional_encoding=False)

        self.assertEqual(value.tolist(), [1])
        self.assertEqual(policy.tolist(), [0, 0, 0, 1, -1, -1, 1, 0, 0, 0])
        self.assertEqual(label.tolist(), [0, 0, 0, 1, 2, 1, 2, 0, 0, 0])


class TestSampleSet(unittest.TestCase):
    def test_fill(self):
        context = Context.standard()
        sample_set = SampleSet()

        initial = Symbol.parse(context, 'a/(x+1)=a-1')
        fits = [FitInfo(1, [0, 0], True), FitInfo(2, [0, 1], False)]
        sample = Sample(initial, fits, useful=True)
        sample_set += sample

        fits = [FitInfo(1, [1, 0], False), FitInfo(2, [1, 1], True)]
        sample = Sample(initial, fits, useful=True)
        sample_set += sample

        self.assertEqual(len(sample_set), 1)

        expected = Sample(
            initial=initial,
            fits=[
                FitInfo(rule_id=1, path=[0, 0], positive=True),
                FitInfo(rule_id=2, path=[0, 1], positive=False),
                FitInfo(rule_id=1, path=[1, 0], positive=False),
                FitInfo(rule_id=2, path=[1, 1], positive=True),
            ],
            useful=True
        )

        self.assertEqual(repr(sample_set.values()[0]), repr(expected))

    def test_merge(self):
        context = Context.standard()
        initial = Symbol.parse(context, 'a/(x+1)=a-1')

        fits = [FitInfo(1, [0, 0], True), FitInfo(2, [0, 1], False)]
        sample = Sample(initial, fits, useful=True)
        set_a = SampleSet()
        set_a += sample

        fits = [FitInfo(1, [1, 0], False), FitInfo(2, [1, 1], True)]
        sample = Sample(initial, fits, useful=True)
        set_b = SampleSet()
        set_b += sample

        set_a.merge(set_b)

        self.assertEqual(len(set_a), 1)

        expected = Sample(
            initial=initial,
            fits=[
                FitInfo(rule_id=1, path=[0, 0], positive=True),
                FitInfo(rule_id=2, path=[0, 1], positive=False),
                FitInfo(rule_id=1, path=[1, 0], positive=False),
                FitInfo(rule_id=2, path=[1, 1], positive=True),
            ],
            useful=True
        )

        self.assertEqual(repr(set_a.values()[0]), repr(expected))

    def test_container(self):
        context = Context.standard()
        sample_set = SampleSet()

        initial = Symbol.parse(context, 'a/(x+1)=a-1')
        fits = [FitInfo(1, [0, 0], True), FitInfo(2, [0, 1], False)]
        sample = Sample(initial, fits, useful=True)
        sample_set += sample

        fits = [FitInfo(1, [1, 0], False), FitInfo(2, [1, 1], True)]
        sample = Sample(initial, fits, useful=True)
        sample_set += sample

        container = sample_set.to_container()
        actual = SampleSet.from_container(container)

        expected = Sample(
            initial=initial,
            fits=[
                FitInfo(rule_id=1, path=[0, 0], positive=True),
                FitInfo(rule_id=2, path=[0, 1], positive=False),
                FitInfo(rule_id=1, path=[1, 0], positive=False),
                FitInfo(rule_id=2, path=[1, 1], positive=True),
            ],
            useful=True
        )

        self.assertEqual(len(actual), 1)
        self.assertEqual(repr(actual.values()[0]), repr(expected))


if __name__ == '__main__':
    unittest.main()
