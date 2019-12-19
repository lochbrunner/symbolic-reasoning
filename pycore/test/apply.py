#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply


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
