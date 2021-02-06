from pycore import Context, Decoration, Symbol, Rule
import unittest


class TestRule(unittest.TestCase):
    def test_rule_new(self):
        context = Context.standard()
        condition = Symbol.parse(context, 'a')
        conclusion = Symbol.parse(context, 'b')
        rule = Rule(condition, conclusion, 'created')

        self.assertEqual(rule.verbose, 'a => b ')

    def test_parse_named(self):
        context = Context.standard()
        rule = Rule.parse(context, 'a => b', 'myrule')
        self.assertEqual(rule.verbose, 'a => b ')
        self.assertEqual(rule.name, 'myrule')

    def test_parse_unnamed(self):
        context = Context.standard()
        rule = Rule.parse(context, 'a => b')
        self.assertEqual(rule.verbose, 'a => b ')
        self.assertEqual(rule.name, 'Parsed from a => b')
