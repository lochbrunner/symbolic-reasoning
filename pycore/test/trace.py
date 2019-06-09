#!/usr/bin/env python3

from itertools import islice
from pycore import Trace, Symbol, Rule, Context

context = Context.standard()

symbol = Symbol.parse(context, "a+b")

print(f'symbol: {symbol}')
print(f'symbol.ident: {symbol.ident}')

print(f'Get arm at [1]: {symbol.get([1])}')

# Traverse symbol
parts = str.join(', ', [part.ident for part in symbol.parts])
print(f'parts: {parts}')
print('')

rule = Rule.parse(context, "a => b")

print(f'rule: {rule}')
print(f'condition: {rule.condition}')
print('')

file = '../../out/trace.bin'

try:
    trace = Trace.load(file)

    for calculation in islice(trace.unroll, 1):
        for step in calculation.steps[0:-1]:
            print(step.deduced)

    for step in islice(trace.all_steps, 2):
        print(step.deduced)
        print(step.path)
        print(step.rule)

    idents = list(trace.meta.used_idents)
    idents_str = str.join(', ', idents)
    print(f'used idents {idents_str}')

    rules = str.join(', ', [str(rule) for rule in trace.meta.rules])
    print(f'rules {rules}')
except Exception as e:
    print(f'Error loading {file}: {e}')
