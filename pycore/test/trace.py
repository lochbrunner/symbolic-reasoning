#!/usr/bin/env python3

from itertools import islice
from pycore import Trace, Symbol, Rule, Context

context = Context.standard()

symbol = Symbol.parse(context, "a+b")

print(symbol)

rule = Rule.parse(context, "a => b")

print(rule)
print(rule.condition)

file = '../../out/trace.bin'

try:
    trace = Trace.load(file)
    print(trace)

    for calculation in islice(trace.unroll(), 1):
        for step in calculation.steps[0:-1]:
            print(step.deduced)

    for step in islice(trace.all_steps(), 5):
        print(step.deduced)
        print(step.path)
        print(step.rule)
except Exception as e:
    print(f'Error loading {file}: {e}')
