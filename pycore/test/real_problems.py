#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply, fit_and_apply, Rule

import unittest


class RealProblems(unittest.TestCase):
    def __init__(self, *args):
        super(RealProblems, self).__init__(*args)
        self.context = Context.standard()

    def variable_creator(self):
        return Symbol.parse(self.context, 'z')

    def assert_step(self, initial: str, target: str, rule_code: str, verbose=False):
        initial_symbol = Symbol.parse(self.context, initial)
        rule = Rule.parse(self.context, rule_code)

        fits = fit_and_apply(self.variable_creator, initial_symbol, rule)
        if verbose:
            deduced = {d.verbose for d, _ in fits}
        else:
            deduced = {str(d) for d, _ in fits}

        self.assertIn(target, deduced)

    # 0 = a + x
    def test_problem_1_step_1(self):
        initial = Symbol.parse(self.context, '0 = a + x')
        rule = Rule.parse(self.context, 'a = b => b = a')

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)

        deduced, mapping = fits[0]

        self.assertEqual(str(deduced), 'a+x=0')
        self.assertEqual(str(mapping.variable[a]), '0')
        self.assertEqual(str(mapping.variable[b]), 'a+x')
        self.assertEqual(mapping.path, [])

    # a + x = 0
    def test_problem_1_step_2(self):
        initial = Symbol.parse(self.context, 'a + x = 0')
        rule = Rule.parse(self.context, 'a+b => b+a')

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)

        deduced, mapping = fits[0]
        self.assertEqual(str(deduced), 'x+a=0')
        self.assertEqual(str(mapping.variable[a]), 'a')
        self.assertEqual(str(mapping.variable[b]), 'x')
        self.assertEqual(mapping.path, [0])

    # x + a = 0
    def test_problem_1_step_3(self):
        initial = Symbol.parse(self.context, 'x + a = 0')
        rule = Rule.parse(self.context, 'a + b = c <=> a = c - b')

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')
        c = Symbol.parse(self.context, 'c')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)
        deduced, mapping = fits[0]

        self.assertEqual(str(deduced), 'x=0-a')
        self.assertEqual(str(mapping.variable[a]), 'x')
        self.assertEqual(str(mapping.variable[b]), 'a')
        self.assertEqual(str(mapping.variable[c]), '0')
        self.assertEqual(mapping.path, [])

    # x = 0 - a
    def test_problem_1_step_4(self):
        initial = Symbol.parse(self.context, 'x = 0 - a')
        rule = Rule.parse(self.context, 'a - b <=> a + (-b)')

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)
        deduced, mapping = fits[0]

        self.assertEqual(str(deduced), 'x=0+-a')
        self.assertEqual(str(mapping.variable[a]), '0')
        self.assertEqual(str(mapping.variable[b]), 'a')
        self.assertEqual(mapping.path, [1])

    # x=0+-a
    def test_problem_1_step_5(self):
        initial = Symbol.parse(self.context, 'x = 0 + -a')
        rule = Rule.parse(self.context, 'a+b => b+a')

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)
        deduced, mapping = fits[0]

        self.assertEqual(str(deduced), 'x=-a+0')
        self.assertEqual(str(mapping.variable[a]), '0')
        self.assertEqual(str(mapping.variable[b]), '-a')
        self.assertEqual(mapping.path, [1])

    # x = -a + 0
    def test_problem_1_step_6(self):
        initial = Symbol.parse(self.context, 'x = -a + 0')
        rule = Rule.parse(self.context, 'a+0 => a')

        a = Symbol.parse(self.context, 'a')

        fits = fit_and_apply(self.variable_creator, initial, rule)
        self.assertEqual(len(fits), 1)
        deduced, mapping = fits[0]

        self.assertEqual(str(deduced), 'x=-a')
        self.assertEqual(str(mapping.variable[a]), '-a')
        self.assertEqual(mapping.path, [1])

    def test_problem_2_step_1(self):
        '''b*x-x=1 -> b*x-1*x=1'''

        initial = 'b*x-x=1'
        target = 'b*x-1*x=1'
        rule = '1*a <= a'
        self.assert_step(initial, target, rule)

    def test_problem_2_step_2(self):

        initial = '(b-1)*x = 1'
        target = 'b-1=1/x'
        rule = 'a*b = c => a = c/b'
        self.assert_step(initial, target, rule)

    def test_problem_3_step_2(self):
        initial = '1+b=x+b'
        target = '1=x+b-b'
        rule = 'a + b = c => a = c - b'
        self.assert_step(initial, target, rule)

    def test_problem_3_step_3(self):
        initial = '1=x+b-b'
        target = '1=x+(b-b)'
        rule = 'a+(b-c) <= (a+b)-c'
        self.assert_step(initial, target, rule, verbose=True)

    def test_problem_3_step_4(self):
        initial = '1=x+(b-b)'
        target = '1=x+0'
        rule = 'a-a => 0'
        self.assert_step(initial, target, rule)

    def test_problem_4_step_1(self):
        initial = 'b*x=a+x-1'
        target = 'b*x=a+x+-1*1'
        rule = 'a-b => a+(-1*b)'

        self.assert_step(initial, target, rule)

    def test_problem_4_step_2(self):
        initial = 'a-1=x*b+-1*x'
        target = 'a-1=x*b+x*-1'
        rule = 'a*b => b*a'

        self.assert_step(initial, target, rule)

    def test_problem_6_step_1(self):
        initial = '(a-b)/x=a/b-1'
        target = '(a-b)/x=(a-1*b)/b'
        rule = 'a/b-c => (a-c*b)/b'

        self.assert_step(initial, target, rule)


if __name__ == '__main__':
    unittest.main()
