from itertools import permutations, islice
from random import shuffle, randint, seed
from copy import deepcopy
import unittest

from torch.utils.data import Dataset

from common.utils import memoize
from deep.dataset.generate import generate_idents
from .symbol_builder import SymbolBuilder


def place_patterns_in_noise(depth=4, spread=2, max_size=120, pattern_depth=2, num_labels=5):
    '''Embeds some unique and fixed patterns (beginning of the alphabet) into noise'''
    if pattern_depth >= depth:
        raise f'Pattern depth ({pattern_depth}) must be smaller than outer depth ({depth})'
    seed(0)

    samples = []

    builder = SymbolBuilder()
    for _ in range(depth):
        builder.add_level_uniform(spread)

    # Noise
    idents_reservoir = generate_idents()

    tree_size = sum([spread**l for l in range(0, depth+1)])
    idents = list(islice(idents_reservoir, tree_size))

    # Create pattern
    idents_reservoir = generate_idents()
    pattern_size = sum([spread**l for l in range(0, pattern_depth+1)])
    if pattern_size*num_labels > tree_size:
        raise Exception(f'Not enough space to place {num_labels} different patterns!')
    patterns = [list(islice(idents_reservoir, pattern_size)) for _ in range(num_labels)]

    max_depth = depth - pattern_depth+1
    pos_count = sum(
        [spread**l for l in range(0, max_depth)])

    label_distribution = [0 for _ in range(0, num_labels+1)]

    max_tries = max_size*10
    while len(samples) < max_size:
        shuffled = deepcopy(idents)
        shuffle(shuffled)

        builder.set_idents_bfs(shuffled)
        for pattern in patterns:
            if builder.has_pattern(pattern):
                continue

        # Find position
        label_id = randint(0, len(patterns)-1)
        pattern = patterns[label_id]
        label_distribution[label_id+1] += 1
        label_distribution[0] += tree_size-1

        pos = randint(0, pos_count-1)
        path = builder.bfs_path(pos)
        builder.set_idents_bfs_at(pattern, path)
        builder.clear_labels()
        # Label 0 means no label
        builder.set_label_at(path, label_id+1)

        samples.append(builder.symbol)

        max_tries -= 1
        if max_tries == 0:
            raise Exception(f'Could not find enough samples.')

    return samples, idents, patterns, label_distribution


class EmbPatternDataset(Dataset):
    '''Embeds different patterns in a noise'''

    def __init__(self, params, transform=None, preprocess=False):
        self.transform = transform
        self.samples, self.idents, self.patterns, self.label_distribution = place_patterns_in_noise(
            depth=params.depth, spread=params.spread, max_size=params.max_size, pattern_depth=1, num_labels=params.num_labels)
        self.preprocess = preprocess
        if preprocess:
            self.samples = [self._process_sample(sample) for sample in self.samples]

    def _process_sample(self, sample):
        x = sample
        s = 2  # spread
        if self.transform is not None:
            return self.transform(x, s)
        return x, s

    def __len__(self):
        return len(self.samples)

    @memoize
    def __getitem__(self, index):
        if not self.preprocess:
            return self._process_sample(self.samples[index])
        else:
            return self.samples[index]

    @property
    def vocab_size(self):
        return len(self.idents)

    @property
    def tag_size(self):
        # One additional for no tag
        return len(self.patterns) + 1

    @property
    def label_weight(self):
        # nodes_count = sum(self.label_distribution)
        min_node = min(self.label_distribution)
        return [min_node/label for label in self.label_distribution]


class TestPatternSegmentation(unittest.TestCase):
    def test_flat(self):
        samples, idents, patterns, _ = place_patterns_in_noise(
            depth=2, spread=2, max_size=12, pattern_depth=1, num_labels=2)

        self.assertCountEqual(idents, ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        self.assertEqual(len(patterns), 2)
        self.assertEqual(len(samples), 12)
        self.assertEqual(samples[0].depth, 2)

        # Check that each sample has at least one pattern
        for sample in samples:
            builder = SymbolBuilder(sample)
            occurred_patterns = sum([builder.has_pattern(pattern) for pattern in patterns])
            self.assertGreaterEqual(occurred_patterns, 1)

        # Check the labels
        for sample in samples:
            builder = SymbolBuilder(sample)
            found = False
            for i, pattern in enumerate(patterns):
                path = builder.find_pattern(pattern)
                if path is not None:
                    actual_label = builder._node_at(path).label
                    if actual_label is not None:
                        self.assertEqual(actual_label, i+1)
                        found = True
                        break
            self.assertTrue(found, 'This sample does not have a pattern')
