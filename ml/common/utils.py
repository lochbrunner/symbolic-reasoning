import logging
import sys
import unittest
from argparse import Namespace
from typing import Dict
from pycore import Scenario, Rule
import torch

logger = logging.getLogger(__name__)


def get_rule_mapping(scenario: Scenario) -> Dict[int, Rule]:
    rule_mapping: Dict[int, Rule] = {}
    used_rules = set()
    max_width = max(len(s.name) for s in scenario.rules)+1
    for i, rule in enumerate(scenario.rules, 1):
        rule_mapping[i] = rule
        used_rules.add(str(rule))
        logger.debug(f'Using rule {i:2}# {rule.name.ljust(max_width)} {rule.verbose}')

    for scenario_rule in scenario.rules:
        if str(scenario_rule) not in used_rules:
            logger.warning(f'The rule "{scenario_rule}" was not in the model created by the training.')
    return rule_mapping


def get_rule_mapping_by_config(config) -> Dict[int, Rule]:
    scenario = Scenario.load(config.files.scenario, no_dependencies=True)
    return get_rule_mapping(scenario)


def make_namespace(value):
    if isinstance(value, dict):
        return Namespace(**{k.replace('-', '_'): make_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [make_namespace(vv) for vv in value]
    return value


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


class Compose(object):
    '''Composes several transforms together.

    Transforms on a generic tuple instead of on value
    as torchvision.transforms.Compose does.
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for t in self.transforms:
            args = t(*args, **kwargs)
        return args


def setup_logging(verbose, log, **kwargs):
    loglevel = 'INFO' if verbose else log.upper()
    if sys.stdin.isatty():
        log_format = '%(message)s'
    else:
        log_format = '%(asctime)s %(message)s'

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )
    # Set the log level of all existing loggers
    all_loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for mod_logger in all_loggers:
        mod_logger._cache.clear()  # pylint: disable=protected-access
        mod_logger.setLevel(logging.getLevelName(loglevel))


def split_dataset(dataset: torch.utils.data.Dataset, validation_ratio=0.1):
    generator = torch.Generator()
    generator.manual_seed(0)
    validation_size = int(len(dataset) * validation_ratio)
    trainings_size = len(dataset) - validation_size
    return torch.utils.data.random_split(dataset, [trainings_size, validation_size], generator=generator)


class TestMakeNamespace(unittest.TestCase):
    def test_recursive(self):
        d = {'a': 'text', 'b': [1, 'c', {'d': 12}]}
        actual = make_namespace(d)

        self.assertEqual(actual.a, 'text')
        self.assertEqual(actual.b[0], 1)
        self.assertEqual(actual.b[1], 'c')
        self.assertEqual(actual.b[2].d, 12)


if __name__ == "__main__":
    unittest.main()
