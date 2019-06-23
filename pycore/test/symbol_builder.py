#!/usr/bin/env python3

from pycore import SymbolBuilder

s = SymbolBuilder()
s.set_level_idents(0, ['a'])

s.add_level_uniform()
s.set_level_idents(1, ['b', 'c'])

print(f'{s} should be a(b, c)')

s.add_level_uniform()
s.set_level_idents(2, ['d', 'e', 'f', 'g'])

print(f'{s} should be a(b(d, e), c(f, g))')
