#!/usr/bin/env python3

from pycore import ScenarioProblems, Rule, Context

import unittest


class TestScenarioProblems(unittest.TestCase):

    def test_io(self):
        filename = '/tmp/test_problems.spb'

        context = Context.standard()
        problems = ScenarioProblems()

        rule = Rule.parse(context, 'a => b')
        problems.add_to_training(rule, 'rule 1')

        rule = Rule.parse(context, 'c => d')
        problems.add_to_validation(rule, 'rule 2')
        problems.add_additional_idents(['a', 'b'])
        problems.add_additional_idents(['b', 'c'])

        problems.dump(filename)

        loaded_problems = ScenarioProblems.load(filename)

        self.assertEqual(len(loaded_problems.validation), 1)
        rule = loaded_problems.validation['rule 2']
        self.assertEqual(rule.verbose.strip(), 'c => d')

        self.assertEqual(len(loaded_problems.training), 1)
        rule = loaded_problems.training['rule 1']
        self.assertEqual(rule.verbose.strip(), 'a => b')

        self.assertCountEqual(loaded_problems.additional_idents, ['a', 'b', 'c'])


if __name__ == '__main__':
    unittest.main()
