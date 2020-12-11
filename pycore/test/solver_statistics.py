#!/usr/bin/env python3

import unittest

from pycore import StepInfo


class TestStepInfo(unittest.TestCase):
    def test_simple_attributes(self):
        step = StepInfo()

        step.current_latex = 'a+b'
        self.assertEqual(step.current_latex, 'a+b')

        step.value = 0.7
        self.assertAlmostEqual(step.value, 0.7)

        step.confidence = 0.2
        self.assertAlmostEqual(step.confidence, 0.2)

        step.rule_id = 4
        self.assertEqual(step.rule_id, 4)

        step.path = [0, 1]
        self.assertEqual(step.path, [0, 1])

        step.top = 1
        self.assertEqual(step.top, 1)

        step.contributed = True
        self.assertEqual(step.contributed, True)

    def test_subsequent(self):
        root = StepInfo()

        self.assertEqual(root.subsequent, [])

        c = StepInfo()
        c.current_latex = 'c'
        root.add_subsequent(c)

        self.assertEqual(root.subsequent, [c])


if __name__ == '__main__':
    unittest.main()
