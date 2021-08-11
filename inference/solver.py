#!/usr/bin/env python3

import pycore


def load_scenario(path):
    pass


class BFS:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end


def main():
    scenario = pycore.Scenario.load('real_world_problems/trigonometric_identities/dataset.yaml')

    # print('-'*40)
    # print('rules')
    # print('-'*40)
    # for name, rule in scenario.rules.items():
    #     print(f'{name}: {rule}')

    # print('-'*40)
    # print('problems')
    # print('-'*40)
    # for name, rule in scenario.problems.items():
    #     print(f'{name}: {rule}')

    # print('-'*40)
    # print('declarations')
    # print('-'*40)
    # for name, rule in scenario.declarations.declarations.items():
    #     print(f'{name}: {rule}')


if __name__ == '__main__':
    main()
