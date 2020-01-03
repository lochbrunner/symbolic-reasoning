#!/usr/bin/env python3

from pycore import Context, Symbol, fit, fit_at

context = Context.standard()

a = Symbol.parse(context, "a+b")
b = Symbol.parse(context, "c+d")

fitmap = fit(a, b)

print(f'fitmap: {fitmap[0]}')

# Fit at
c = Symbol.parse(context, "3*(e+f)")
fitmap = fit_at(c, a, [1])
print(f'fitmap: {fitmap[0]}')
