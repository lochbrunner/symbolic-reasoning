#!/usr/bin/env python3

from itertools import islice
from pycore import Trace, Rule, Context

context = Context.standard()

rule = Rule.parse(context, "a => b")

print(f'rule: {rule}')
print(f'condition: {rule.condition}')
print('')

file = '../../out/generator/trace.bin'

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
