#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply, fit_and_apply, fit_at_and_apply, Rule

import unittest


class TestApply(unittest.TestCase):

    def __init__(self, *args):
        super(TestApply, self).__init__(*args)
        self.context = Context.standard()

    def variable_creator(self):
        return Symbol.parse(self.context, 'z')

    def test_readme_example(self):

        # First step
        initial = Symbol.parse(self.context, 'b*(c*d-c*d)=e')
        # Rule
        condition = Symbol.parse(self.context, 'a-a')
        conclusion = Symbol.parse(self.context, '0')

        mapping = fit(initial, condition)

        deduced = apply(mapping[0], self.variable_creator, initial, conclusion)

        self.assertEqual(str(deduced), 'b*0=e')

        # Second step

        # Rule
        condition = Symbol.parse(self.context, 'a*0')
        conclusion = Symbol.parse(self.context, '0')

        mapping = fit(deduced, condition)

        deduced = apply(mapping[0], self.variable_creator, deduced, conclusion)

        self.assertEqual(str(deduced), '0=e')

    def test_fit_and_apply(self):
        initial = Symbol.parse(self.context, 'b*(c*d-c*d)=e')
        # All in once
        rule = Rule.parse(self.context, 'a-a => 0')
        deduced, mapping = fit_and_apply(self.variable_creator, initial, rule)[0]
        self.assertEqual(str(deduced), 'b*0=e')
        self.assertEqual(mapping.path, [0, 1])
        orig, target = list(mapping.variable.items())[0]
        self.assertEqual(str(orig), 'a')
        self.assertEqual(str(target), 'c*d')

    def test_fit_at_and_apply(self):
        initial = Symbol.parse(self.context, 'b*(c*d-c*d)=e')
        rule = Rule.parse(self.context, 'a-a => 0')
        s, m = fit_at_and_apply(self.variable_creator, initial, rule, [0, 1])
        self.assertEqual(m.path, [0, 1])
        orig, target = list(m.variable.items())[0]
        self.assertEqual(str(orig), 'a')
        self.assertEqual(str(target), 'c*d')
        self.assertEqual(str(s), 'b*0=e')

    def test_next_exponent(self):
        initial = Symbol.parse(self.context, 'x*x^1')
        rule = Rule.parse(self.context, 'a*a^n => a^(n+1)')
        deduced, _ = fit_and_apply(self.variable_creator, initial, rule)[0]
        self.assertEqual(str(deduced), 'x^(1+1)')
