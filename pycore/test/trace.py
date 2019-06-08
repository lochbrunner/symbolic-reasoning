#!/usr/bin/env python

from itertools import islice
from pycore import Trace, Symbol, Rule, Context

context = Context.standard()

symbol = Symbol.parse(context, "a+b")

print(symbol)

rule = Rule.parse(context, "a => b")

print(rule)
print(rule.condition)


trace = Trace.load("../../out/trace.bin")
print(trace)

for calculation in islice(trace.unroll(), 1):
    for step in calculation.steps[0:-1]:
        print(step.deduced)
