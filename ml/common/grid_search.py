from functools import reduce
import operator
import random
from argparse import Namespace

import unittest


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class Range:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)


class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class MonteCarloSampling:
    def __init__(self, size, seed=0):
        self.size = size
        self.rng = random.Random(seed)

    def sample(self, values):
        return self.rng.sample(values, self.size)


class Unroller:
    def __init__(self, args):
        self.ranges = []
        self.constants = []
        self.use_namespace = isinstance(args, Namespace)
        if self.use_namespace:
            args = vars(args)

        for (k, v) in args.items():
            if isinstance(v, list):
                self.ranges.append(Range(k, v))
            else:
                self.constants.append(Constant(k, v))

        self.combination_count = prod([len(range) for range in self.ranges])

    def __len__(self):
        return self.combination_count

    def __getitem__(self, i):
        i = i % self.combination_count
        arg = {}
        for (j, r) in enumerate(self.ranges):
            prev = prod([len(range) for range in self.ranges[:j]])
            index = (i // prev) % len(r)
            arg[r.name] = r.values[index]
        for c in self.constants:
            arg[c.name] = c.value
        if self.use_namespace:
            return Namespace(**arg)
        else:
            return arg


def unroll(args, sampling=None):
    '''
    Unrolls all combinations of values in the dict,
    while multiply of the array values.
    '''

    unroller = Unroller(args)

    samples = range(len(unroller))

    if callable(getattr(sampling, 'sample', None)):
        samples = sampling.sample(samples)

    for i in samples:
        yield unroller[i]


def unroll_many(*args, sampling=None):
    unrollers = [Unroller(arg) for arg in args]
    combination_count = prod([len(unroller) for unroller in unrollers])

    samples = range(combination_count)

    if callable(getattr(sampling, 'sample', None)):
        samples = sampling.sample(samples)

    for i in samples:
        yield [unroller[i//prod([len(unroller) for unroller in unrollers[:j]])]
               for j, unroller in enumerate(unrollers)]


def _to_key_value(arg):
    if isinstance(arg, Namespace):
        return vars(arg).items()
    else:
        return arg.items()


def get_range_names(*args):
    return [k for arg in args
            for (k, v) in _to_key_value(arg)
            if isinstance(v, list)]


def strip_keys(*args, names):
    return {k: v for arg in args for k, v in _to_key_value(arg) if k in names}


class TestUnroll(unittest.TestCase):
    def test_dictionary_full(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}

        unrolled = list(unroll(A))

        self.assertListEqual(unrolled,
                             [{'a': 1, 'b': 4, 'c': 6},
                              {'a': 2, 'b': 4, 'c': 6},
                              {'a': 1, 'b': 5, 'c': 6},
                              {'a': 2, 'b': 5, 'c': 6}]
                             )

    def test_dictionary_random(self):
        A = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': 6}

        size = 4
        unrolled = list(unroll(A, MonteCarloSampling(size)))

        self.assertEqual(len(unrolled), size)

    def test_namespace_full(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}
        A = Namespace(**A)

        unrolled = list(unroll(A))

        self.assertListEqual(unrolled,
                             [Namespace(a=1, b=4, c=6),
                              Namespace(a=2, b=4, c=6),
                              Namespace(a=1, b=5, c=6),
                              Namespace(a=2, b=5, c=6)]
                             )

    def test_multiply(self):
        A = {'a': [1, 2]}
        B = {'b': [3, 4]}

        unrolled = list(unroll_many(A, B))

        self.assertListEqual(unrolled,
                             [[{'a': 1}, {'b': 3}],
                              [{'a': 2}, {'b': 3}],
                              [{'a': 1}, {'b': 4}],
                              [{'a': 2}, {'b': 4}]]
                             )

    def test_get_ranges_single(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}
        names = get_range_names(A)
        self.assertListEqual(names, ['a', 'b'])

    def test_get_ranges_multiple(self):
        A = {'a': [1, 2]}
        B = {'b': [3, 4]}
        names = get_range_names(A, B)
        self.assertListEqual(names, ['a', 'b'])

    def test_get_ranges_single_namespace(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}
        A = Namespace(**A)
        names = get_range_names(A)
        self.assertListEqual(names, ['a', 'b'])

    def test_strip_keys(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}
        D = {'d': [3, 4]}
        names = get_range_names(A, D)

        A = {'a': 1, 'b': 5, 'c': 6}
        D = {'d': 3}
        merged_dicts = strip_keys(A, D, names=names)

        self.assertEqual(merged_dicts, {'a': 1, 'b': 5, 'd': 3})
