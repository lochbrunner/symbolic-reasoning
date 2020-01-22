#!/usr/bin/env python3

from pycore import Bag

import unittest


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


if __name__ == '__main__':
    unittest.main()
