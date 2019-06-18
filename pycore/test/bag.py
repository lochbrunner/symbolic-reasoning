#!/usr/bin/env python3

from pycore import Bag

bag = Bag.load('../../out/generator/bag-4-4.bin')

# Meta
print(f'idents: {str.join(", ",bag.meta.idents)}')

for stat in bag.meta.rules:
    print(f'Rule: {stat.rule}  [{stat.fits}]')

rule_to_ix = {str(rule): i for i, rule in enumerate(bag.meta.rules)}
print(f'number od rules (dict) {len(rule_to_ix)}')
print(f'number od rules (orig) {len(bag.meta.rules)}')

# Samples
print(f'Number of samples: {len(bag.samples)}')

for sample in bag.samples[:3]:
    print(f'Initial {sample.initial}')
    for fit in sample.fits:
        print(f'rule: {fit.rule} @{fit.path}')
