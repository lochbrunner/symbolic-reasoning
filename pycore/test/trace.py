#!/usr/bin/env python

from pycore import Trace, Symbol, Rule, Context

context = Context.standard()

symbol = Symbol.parse(context, "a+b")

print(symbol)

rule = Rule.parse(context, "a => b")

print(rule)
print(rule.condition)


trace = Trace.load("../../out/trace.bin")
print(trace)

for calculation in trace.unroll():
    for step in calculation.steps:
        print(step.deduced)
