from itertools import islice
from string import ascii_lowercase as alphabet
from random import shuffle, randint
from copy import deepcopy
import unittest

from typing import List, Set, Dict, Tuple, Optional

from .symbol_builder import SymbolBuilder


def generate_idents():
    # First use one char of alpahabet
    for c in alphabet:
        yield c

    for c in alphabet:
        for cc in alphabet:
            yield c + cc


def scenarios_choices():
    return ['permutation', 'pattern']


def create_complex_pattern_in_noise(depth=4, spread=2, max_size=120, pattern_depth=1):
    '''Embeds a unique and fixed pattern (beginning of the alphabet) into noise'''

    if pattern_depth >= depth:
        raise f'Pattern depth ({pattern_depth}) must be smaller than outer depth ({depth})'
    samples = []

    builder = SymbolBuilder()
    for _ in range(depth-1):
        builder.add_level_uniform(spread)

    # Create pattern
    idents_reservoir = generate_idents()
    pattern_size = sum([spread**l for l in range(0, pattern_depth)])
    pattern = list(islice(idents_reservoir, pattern_size))
    # pattern_builder = SymbolBuilder()
    # for _ in range(pattern_depth):
    #     builder.add_level_uniform(spread)
    # pattern_size = sum([spread**l for l in range(0, pattern_depth+1)])
    # idents = list(islice(idents_reservoir, pattern_size))
    # pattern_builder.set_idents_bfs(idents)
    # pattern = pattern_builder.symbol

    # Noise
    idents_reservoir = generate_idents()

    size = sum([spread**l for l in range(0, depth)])
    idents = list(islice(idents_reservoir, size))
    classes = []

    max_depth = depth - pattern_depth+1
    pos_count = pattern_size = sum(
        [spread**l for l in range(0, max_depth)])

    classes_dict = {}

    while len(samples) < max_size:
        shuffled = deepcopy(idents)
        shuffle(shuffled)

        builder.set_idents_bfs(shuffled)
        if builder.has_pattern(pattern):
            continue
        # Find position
        pos = randint(0, pos_count-1)
        path = builder.bfs_path(pos)
        builder.set_idents_bfs_at(pattern, path)

        if pos not in classes_dict:
            classes_dict[pos] = len(classes_dict)
            classes.append(classes_dict[pos])
        class_id = classes_dict[pos]

        samples.append((class_id, builder.symbol))

    return samples, idents, classes


class TestSinglePattern(unittest.TestCase):
    def test_slim_pattern(self):
        samples, idents, classes = create_complex_pattern_in_noise(
            depth=3, spread=1, max_size=2, pattern_depth=2)

        builder = SymbolBuilder()

        builder.childs = [samples[0][1]]
        self.assertTrue(builder.has_pattern(['a', 'b']))

        builder.childs = [samples[1][1]]
        self.assertTrue(builder.has_pattern(['a', 'b']))

        self.assertEqual(samples[0][1].depth, 2)
        self.assertEqual(samples[1][1].depth, 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertGreaterEqual(len(classes), 1)

    def test_wide_pattern(self):
        samples, idents, classes = create_complex_pattern_in_noise(
            depth=3, spread=2, max_size=2, pattern_depth=2)

        builder = SymbolBuilder()

        builder.childs = [samples[0][1]]
        self.assertTrue(builder.has_pattern(['a', 'b', 'c']))

        builder.childs = [samples[1][1]]
        self.assertTrue(builder.has_pattern(['a', 'b', 'c']))

        self.assertEqual(samples[0][1].depth, 2)
        self.assertEqual(samples[1][1].depth, 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(idents, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        self.assertGreaterEqual(len(classes), 1)
