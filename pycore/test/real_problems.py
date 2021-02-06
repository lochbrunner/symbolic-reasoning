#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply, fit_and_apply, Rule

import unittest


class RealProblems(unittest.TestCase):
    def __init__(self, *args):
        super(RealProblems, self).__init__(*args)
        self.context = Context.standard()

    def variable_creator(self):
        return Symbol.parse(self.context, 'z')

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


if __name__ == '__main__':
    unittest.main()
