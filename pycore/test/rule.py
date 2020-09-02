from pycore import Context, Decoration, Symbol, Rule
import unittest


class TestRule(unittest.TestCase):
    def test_rule_new(self):
        context = Context.standard()
        condition = Symbol.parse(context, 'a')
        conclusion = Symbol.parse(context, 'b')
        rule = Rule(condition, conclusion, 'created')

        self.assertEqual(rule.verbose, 'a => b ')
