#!/usr/bin/env python3

from pycore import Context, Symbol

from typing import List

context = Context.standard()

a = Symbol.parse(context, "a+b")


def unroll(x: Symbol) -> List[Symbol]:
    stack = [x]
    x = []
    seen = set()
    while len(stack) > 0:
        n = stack[-1]
        if len(n.childs) > 0 and n not in seen:
            stack.extend(n.childs[::-1])
            seen.add(n)
        else:
            stack.pop()
            yield n


for n in unroll(a):
    print(f'n: {n}')

leaf = Symbol.parse(context, "c")

a = Symbol.parse(context, "a")
a.pad('<PAD>', 2, 1)
childs = ', '.join([str(c.ident) for c in a.childs])
print(f'a.childs: {childs}')


# Test
print('Test')
a = Symbol.parse(context, 'x=0*x/1')
a.pad('<PAD>', 2, 2)
print(a.tree)
