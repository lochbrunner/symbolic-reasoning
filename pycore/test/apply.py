#!/usr/bin/env python3

from pycore import Context, Symbol, fit, apply

context = Context.standard()


def variable_creator():
    return Symbol.parse(context, "z")


initial = Symbol.parse(context, "b*(c*d-c*d)=e")
condition = Symbol.parse(context, "a-a")
conclusion = Symbol.parse(context, "0")

mapping = fit(initial, condition)


deduced = apply(mapping[0], variable_creator, initial, conclusion)

print(deduced)
