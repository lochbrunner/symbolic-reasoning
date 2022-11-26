#!/usr/bin/env python3

import unittest

from pycore import Context, Symbol, fit, fit_at


class TestFit(unittest.TestCase):
    def __init__(self, *args):
        super(TestFit, self).__init__(*args)
        self.context = Context.standard()

    def test_fit(self):
        s1 = Symbol.parse(self.context, 'a+b')
        s2 = Symbol.parse(self.context, 'c+d')

        fitmap = fit(s1, s2)[0]

        a = Symbol.parse(self.context, 'a')
        b = Symbol.parse(self.context, 'b')
        c = Symbol.parse(self.context, 'c')
        d = Symbol.parse(self.context, 'd')

        self.assertEqual(fitmap.path, [])
        self.assertEqual(fitmap.variable[c], a)
        self.assertEqual(fitmap.variable[d], b)

    def test_fit_at(self):
        a = Symbol.parse(self.context, 'a+b')
        c = Symbol.parse(self.context, '3*(e+f)')

        fitmap = fit_at(c, a, [1])
        if fitmap is None:
            raise AssertionError()
        self.assertIsNotNone(fitmap)
        self.assertEqual(fitmap.path, [1])
        a = Symbol.parse(self.context, 'a')
        e = Symbol.parse(self.context, 'e')
        b = Symbol.parse(self.context, 'b')
        f = Symbol.parse(self.context, 'f')

        self.assertEqual(fitmap.variable[a], e)
        self.assertEqual(fitmap.variable[b], f)

    def test_fit_at_none(self):
        a = Symbol.parse(self.context, 'a+b')
        c = Symbol.parse(self.context, '3*(e+f)')

        fitmap = fit_at(c, a, [0, 2])
        self.assertIsNone(fitmap)
