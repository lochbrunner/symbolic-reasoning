#!/usr/bin/env python3

from pycore import Bag

bag = Bag.load('../../out/generator/bag-1.bin')

# Meta
print(f'idents: {str.join(", ",bag.meta.idents)}')

for stat in bag.meta.rules[:3]:
    print(f'Rule: {stat.rule}  [{stat.fits}]')

# Samples
print(f'Number of samples: {len(bag.samples)}')

for sample in bag.samples[:3]:
    print(f'Initial {sample.initial}')
    for fit in sample.fits:
        print(f'rule: {fit.rule} @{fit.path}')
