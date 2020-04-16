from functools import reduce
import operator
import random

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


def unroll(args, sampling=None):
    '''
    Unrolls all combinations of values in the dict,
    while multiply of the array values.
    '''
    ranges = []
    constants = []

    for (k, v) in args.items():
        if isinstance(v, list):
            ranges.append(Range(k, v))
        else:
            constants.append(Constant(k, v))

    combination_count = prod([len(range) for range in ranges])
    samples = range(combination_count)

    if callable(getattr(sampling, 'sample', None)):
        samples = sampling.sample(samples)

    for i in samples:
        arg = {}
        for (j, r) in enumerate(ranges):
            prev = prod([len(range) for range in ranges[:j]])
            index = (i // prev) % len(r)
            arg[r.name] = r.values[index]
        for c in constants:
            arg[c.name] = c.value
        yield arg


class TestUnroll(unittest.TestCase):
    def test_unroll_full(self):
        A = {'a': [1, 2], 'b': [4, 5], 'c': 6}

        unrolled = list(unroll(A))

        self.assertListEqual(unrolled,
                             [{'a': 1, 'b': 4, 'c': 6},
                              {'a': 2, 'b': 4, 'c': 6},
                              {'a': 1, 'b': 5, 'c': 6},
                              {'a': 2, 'b': 5, 'c': 6}]
                             )

    def test_unroll_random(self):
        A = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': 6}

        size = 4

        unrolled = list(unroll(A, MonteCarloSampling(size)))

        self.assertEqual(len(unrolled), size)
