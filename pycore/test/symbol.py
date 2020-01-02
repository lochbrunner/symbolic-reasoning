#!/usr/bin/env python3

from pycore import Context, Symbol

from typing import List

context = Context.standard()

a = Symbol.parse(context, 'a+b')


def unroll(x: Symbol) -> List[Symbol]:
    stack = [(x, '/')]
    x = []
    # Using the path as unique id because id(n) is not unique enough ;)
    # id(n) is the memory address which get reused by the rust backend.
    seen = set()
    while len(stack) > 0:
        n, p = stack[-1]
        if len(n.childs) > 0 and p not in seen:
            stack.extend([(c, f'{p}{i}') for i, c in enumerate(n.childs[::-1])])
            seen.add(p)
        else:
            stack.pop()
            yield n


for n in unroll(a):
    print(f'n: {n}')

leaf = Symbol.parse(context, 'c')

a = Symbol.parse(context, 'a')
a.pad('<PAD>', 2, 1)
childs = ', '.join([str(c.ident) for c in a.childs])
print(f'a.childs: {childs}')


# Test Padding
print('Test padding ...')
a = Symbol.parse(context, 'x=0*x/1')
a = a.create_padded('<PAD>', 2, 5)
padded_size = a.tree.count('\n')
unrolled = unroll(a)
exprected_len = sum(2**l for l in range(6))
print(f'unrolled: {len(list(unrolled))} (expected: {exprected_len})')
# print(a.tree)

# Test comparison
print('Test comparison ...')
a = Symbol.parse(context, 'a+c')
b = Symbol.parse(context, 'a+b')
a1 = Symbol.parse(context, 'a+c')
print(f'eq: {a==b}')
print(f'eq: {a==a1}')

print(f'hash of {a}: {hash(a)} <{id(a)}>')
print(f'hash of {a1}: {hash(a1)} <{id(a1)}>')
print(f'hash of {b}: {hash(b)} <{id(b)}>')

# Test comparison
print('Attribute ...')
a.label = 'blub'

print(f'a.label: {a.label}')
del a.label
try:
    print(f'a.label: {a.label}')
except KeyError as e:
    print(e)


# Test comparison
print('Traverse ...')
a = Symbol.parse(context, 'a*b+c*d')
print(f'dfs {a}')
for part in a.parts_dfs:
    print(part)

print(f'bfs {a}')
for part in a.parts_bfs:
    print(part)
