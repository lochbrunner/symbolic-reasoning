#!/usr/bin/env python3

from pycore import Context, Symbol, fit

context = Context.standard()

a = Symbol.parse(context, "a+b")
b = Symbol.parse(context, "c+d")

fitmap = fit(a, b)

print(f'fitmap: {fitmap}')
