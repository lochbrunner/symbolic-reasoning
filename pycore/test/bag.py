#!/usr/bin/env python3

from pycore import Bag, Sample, Symbol, Context, FitInfo, Container, Rule, SampleSet

import unittest
import numpy.testing as npt


@unittest.skip('No bagfile available')
class TestBag(unittest.TestCase):

    def test_bag(self):

        bag = Bag.load('./../../out/generator/bag-2-2.bin')

        # Meta
        self.assertGreater(len(bag.meta.idents), 0)
        # print(f'idents: {str.join(", ",bag.meta.idents)}')

        self.assertGreater(len(bag.meta.rules), 0)
        # rules = [str(rule) for rule in bag.meta.rules]
        # print(f'rules: {rules}')

        self.assertGreater(len(bag.meta.rule_distribution), 1)
        # print(f'rule distribution: {bag.meta.rule_distribution}')

        # Samples
        self.assertGreater(len(bag.samples), 0)
        # print(f'Number of containers: {len(bag.samples)}')
        self.assertIsNotNone(bag.samples[-1].max_spread)
        self.assertIsNotNone(bag.samples[-1].max_depth)
        # print(f'Size of last: s: {bag.samples[-1].max_spread} d: {bag.samples[-1].max_depth}')

        sample = bag.samples[-1].samples[0]
        # print(f'Sample {sample.initial} ({len(sample.fits)} fits)')
        # path = [str(path) for path in sample.fits[0].path]
        # path = '/'.join(path)
        # rule = bag.meta.rules[sample.fits[0].rule]
        # print(f'Fits: {rule.name} @ {path}')

        # Testing label
        a = sample.initial
        a.label = 12

        b = a
        # for rule in bag.meta.rules:
        #     print(f'rule {rule.name}: {rule.condition}')

        self.assertEqual(b.label, 12)
        a.label = 15
        self.assertEqual(b.label, 15)


class TestSample(unittest.TestCase):

    def test_create_sample(self):
        context = Context.standard()
        symbol = Symbol.parse(context, "a")
        fits = [FitInfo(3, [1, 2], True), FitInfo(7, [3, 4], False)]
        sample = Sample(symbol, fits)

        self.assertEqual(sample.initial.verbose, 'a')
        # Positive fit
        self.assertEqual(sample.fits[0].policy, 1.)
        self.assertEqual(sample.fits[0].path, [1, 2])
        self.assertEqual(sample.fits[0].rule, 3)
        # Negative fit
        self.assertEqual(sample.fits[1].policy, -1.)
        self.assertEqual(sample.fits[1].path, [3, 4])
        self.assertEqual(sample.fits[1].rule, 7)

    def test_create_bag(self):
        context = Context.standard()
        symbol = Symbol.parse(context, "a")
        fits = [FitInfo(1, [1, 2], True), FitInfo(1, [3, 4], False)]
        sample = Sample(symbol, fits)
        container = Container()

        container.add_sample(sample)

        rule = Rule.parse(context, 'a => b')
        padding = Rule.parse(context, 'a => a')

        bag = Bag(rules=[('padding', padding), ('Test', rule)])
        bag.add_container(container)
        bag.update_meta()

        self.assertEqual(bag.meta.idents, ['a'])
        self.assertEqual(bag.meta.rule_distribution, [(0, 0), (1, 1)])
        self.assertEqual(bag.containers[0].samples[0].initial.verbose, 'a')

    def test_embed(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b=c*d')
        embed_dict = {'=': 1, '+': 2, '*': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7}
        fits = [FitInfo(2, [0], True)]
        useful = True
        index_encoding = True
        positional_encoding = False
        sample = Sample(symbol, fits, useful)
        spread = 2
        max_depth = symbol.depth
        embedding, indices, positional_encoding, label, policy, value = sample.embed(
            embed_dict, 0, spread, max_depth, index_encoding, positional_encoding)

        self.assertIsNone(positional_encoding)
        npt.assert_equal(embedding[:, 0], [1, 2, 3, 4, 5, 6, 7, 0])
        npt.assert_equal(embedding[:, 1], [1, 1, 1, 0, 0, 0, 0, 0])  # is operator
        npt.assert_equal(embedding[:, 2], [1, 1, 1, 0, 0, 0, 0, 0])  # is fixed
        npt.assert_equal(embedding[:, 3], [0, 0, 0, 0, 0, 0, 0, 0])  # is number
        npt.assert_equal(indices, [[0, 1, 2, 7],
                                   [1, 3, 4, 0],
                                   [2, 5, 6, 0],
                                   [3, 7, 7, 1],
                                   [4, 7, 7, 1],
                                   [5, 7, 7, 2],
                                   [6, 7, 7, 2],
                                   [7, 7, 7, 7]])
        npt.assert_equal(label, [0, 2, 0, 0, 0, 0, 0, 0])
        npt.assert_equal(policy, [0, 1., 0, 0, 0, 0, 0, 0])
        npt.assert_equal(value, [1])


class TestSampleSet(unittest.TestCase):
    def test_two_samples(self):
        context = Context.standard()
        symbol_a = Symbol.parse(context, "a")
        symbol_b = Symbol.parse(context, "b")
        sample_set = SampleSet()
        sample_set.add(Sample(symbol_a, [FitInfo(0, [1, 2], True)]))
        sample_set.add(Sample(symbol_a, [FitInfo(1, [1, 2], True)]))
        sample_set.add(Sample(symbol_b, [FitInfo(0, [1, 2], True)]))

        self.assertEqual(len(sample_set), 2)

        container = sample_set.to_container()
        self.assertEqual(len(container), 2)

    def test_multiple_fits(self):
        context = Context.standard()
        symbol = Symbol.parse(context, "a")
        sample_set = SampleSet()
        sample_set.add(Sample(symbol, [FitInfo(0, [1, 2], True)]))
        sample_set.add(Sample(symbol, [FitInfo(1, [1, 2], True)]))
        sample_set.add(Sample(symbol, [FitInfo(0, [1, 2], False)]))
        sample_set.add(Sample(symbol, [FitInfo(0, [1, 3], False)]))

        self.assertEqual(len(sample_set), 1)

        container = sample_set.to_container()
        sample = container.samples[0]

        self.assertEqual(sample.initial.verbose, 'a')
        self.assertEqual(len(sample.fits), 3)

        self.assertEqual(FitInfo(0, [1, 2], True), FitInfo(0, [1, 2], True))

        # Expect to 3. sample to be missing
        self.assertCountEqual(sample.fits, [FitInfo(0, [1, 2], True),
                                            FitInfo(1, [1, 2], True),
                                            FitInfo(0, [1, 3], False)])


if __name__ == '__main__':
    unittest.main()
