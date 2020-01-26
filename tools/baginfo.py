#!/usr/bin/env python3

from pycore import Bag

import argparse


def print_info(name, value, c1, c2):
    name = name.ljust(c1)
    value = str(value).rjust(c2)
    print(f'{name} {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('baginfo')
    parser.add_argument('filename')
    args = parser.parse_args()

    bag = Bag.load(args.filename)

    c1 = max([len(rule.name) for rule in bag.meta.rules])+2
    c2 = max(len(str(dist)) for dist in bag.meta.rule_distribution)

    rules_tuple = list(zip(bag.meta.rules, bag.meta.rule_distribution))
    rules_tuple.sort(key=lambda r: r[1], reverse=True)

    for (rule, count) in rules_tuple:
        print_info(f'{rule.name}:', count, c1, c2)

    print('-'*(c1+c2+1))
    total = sum(bag.meta.rule_distribution[1:])
    print_info('total:', total, c1, c2)
    print_info('idents:', len(bag.meta.idents), c1, c2)
