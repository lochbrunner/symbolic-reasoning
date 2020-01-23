#!/usr/bin/env python3

from pycore import Bag

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('baginfo')
    parser.add_argument('filename')
    args = parser.parse_args()

    bag = Bag.load(args.filename)

    max_name_length = max([len(rule.name) for rule in bag.meta.rules])+2
    max_count_length = max(len(str(dist)) for dist in bag.meta.rule_distribution)

    rules_tuple = list(zip(bag.meta.rules, bag.meta.rule_distribution))
    rules_tuple.sort(key=lambda r: r[1], reverse=True)

    for (rule, count) in rules_tuple:
        rule_name = f'{rule.name}:'.ljust(max_name_length)
        count = str(count).rjust(max_count_length)
        print(f'{rule_name} {count}')

    print('-'*(max_name_length+max_count_length+1))
    total = sum(bag.meta.rule_distribution[1:])
    rule_name = 'total:'.ljust(max_name_length)
    count = str(total).rjust(max_count_length)
    print(f'{rule_name} {count}')
