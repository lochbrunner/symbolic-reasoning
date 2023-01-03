#!/usr/bin/env python3

import unittest

from pycore import SymbolBuilder


class TestSymbolBuilder(unittest.TestCase):
    def test_symbol_builder(self):

        s = SymbolBuilder()
        s.set_level_idents(0, ['a'])

        s.add_level_uniform()
        s.set_level_idents(1, ['b', 'c'])

        self.assertEqual(str(s), 'a(b, c)')

        s.add_level_uniform()
        s.set_level_idents(2, ['d', 'e', 'f', 'g'])

        self.assertEqual(str(s), 'a(b(d, e), c(f, g))')
