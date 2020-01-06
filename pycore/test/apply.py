#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply, fit_and_apply, fit_at_and_apply, Rule


# Readme example
context = Context.standard()


def variable_creator():
    return Symbol.parse(context, "z")


# First step
initial = Symbol.parse(context, "b*(c*d-c*d)=e")
# Rule
condition = Symbol.parse(context, "a-a")
conclusion = Symbol.parse(context, "0")

mapping = fit(initial, condition)

deduced = apply(mapping[0], variable_creator, initial, conclusion)

# >> b*0=e
print(f'Deduced: {deduced}')

# Second step

# Rule
condition = Symbol.parse(context, "a*0")
conclusion = Symbol.parse(context, "0")

mapping = fit(deduced, condition)

deduced = apply(mapping[0], variable_creator, initial, conclusion)

# >> b*0=e
print(f'Deduced: {deduced}')


# All in once
rule = Rule.parse(context, "a-a => 0")
deduced, mapping = fit_and_apply(variable_creator, initial, rule)[0]
print(f'Deduced {deduced} with {mapping}')

s, m = fit_at_and_apply(variable_creator, initial, rule, [0, 1])
print(f'Deduced at: {s} with {m}')
