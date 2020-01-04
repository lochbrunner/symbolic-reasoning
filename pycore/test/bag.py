#!/usr/bin/env python3

from pycore import Bag

bag = Bag.load('./out/generator/bag-2-2.bin')

# Meta
print(f'idents: {str.join(", ",bag.meta.idents)}')

rules = [str(rule) for rule in bag.meta.rules]
print(f'rules: {rules}')

print(f'rule distribution: {bag.meta.rule_distribution}')

# Samples
print(f'Number of containers: {len(bag.samples)}')
print(f'Size of last: s: {bag.samples[-1].max_spread} d: {bag.samples[-1].max_depth}')

sample = bag.samples[-1].samples[0]
print(f'Sample {sample.initial} ({len(sample.fits)} fits)')
path = [str(path) for path in sample.fits[0].path]
path = '/'.join(path)
rule = bag.meta.rules[sample.fits[0].rule]
print(f'Fits: {rule.name} @ {path}')

# Testing label
a = sample.initial
a.label = 12

b = a

print(f'b: {b.label}')
a.label = 15
print(f'b: {b.label}')
