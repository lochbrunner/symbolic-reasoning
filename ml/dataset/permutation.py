from itertools import permutations, islice
import unittest

from torch.utils.data import Dataset

from common.utils import memoize
from common.node import Node
from .generate import generate_idents
from .symbol_builder import SymbolBuilder


def create_samples_permutation(depth=2, spread=1, max_size=120):
    '''
    Samples are of the form (class_id, symbol)
    Using a permutation where of unique idents
    '''
    samples = []

    size = sum([spread**l for l in range(0, depth+1)])

    builder = SymbolBuilder()
    for _ in range(depth):
        builder.add_level_uniform(spread)
    # This line is needed because of python bug while running the unit tests.
    builder.clear_labels()

    idents = list(islice(generate_idents(), size))
    classes = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)

        builder.set_idents_bfs(perm)

        samples.append((i, builder.symbol))

        if len(samples) >= max_size:
            break

    return samples, idents, classes


class PermutationDataset(Dataset):
    '''Deprecated!'''

    def __init__(self, params, transform=None):
        self.transform = transform
        self.samples, self.idents, self.classes = create_samples_permutation(
            depth=params.depth, spread=params.spread)

        # Preprocess
        self.samples = [self._process_sample(sample) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    @memoize
    def __getitem__(self, index):
        return self.samples[index]
        # return _process_sample(index)

    def _process_sample(self, sample):
        [y, x] = sample
        s = 2  # spread
        if self.transform is not None:
            return self.transform(x, y, s)
        return x, y, s

    @property
    def vocab_size(self):
        return len(self.idents)

    @property
    def tag_size(self):
        return len(self.classes)

    @property
    def label_weight(self):
        return [1 for _ in range(self.tag_size)]


class TestPermutation(unittest.TestCase):
    def test_flat(self):
        samples, idents, classes = create_samples_permutation(
            depth=1, spread=1)

        self.assertEqual(idents, ['a', 'b'])
        self.assertEqual(classes, [0, 1])

        self.assertEqual(len(samples), 2)

        expected = Node()
        expected.ident = 'a'
        expected.childs = [Node('b')]
        self.assertEqual(expected, samples[0][1])

        expected = Node()
        expected.ident = 'b'
        expected.childs = [Node('a')]
        self.assertEqual(expected, samples[1][1])

    def test_wide(self):
        samples, idents, classes = create_samples_permutation(
            depth=1, spread=2)

        self.assertEqual(len(samples), 6)

        expected_samples = []
        expected_samples.append(Node('a', [Node('b'), Node('c')]))
        expected_samples.append(Node('a', [Node('c'), Node('b')]))
        expected_samples.append(Node('b', [Node('a'), Node('c')]))
        expected_samples.append(Node('b', [Node('c'), Node('a')]))
        expected_samples.append(Node('c', [Node('a'), Node('b')]))
        expected_samples.append(Node('c', [Node('b'), Node('a')]))

        self.maxDiff = None
        self.assertCountEqual([sample[1]
                               for sample in samples], expected_samples)

        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertEqual(classes, [0, 1, 2, 3, 4, 5])

    def test_deep(self):
        samples, idents, classes = create_samples_permutation(
            depth=2, spread=1)

        self.assertEqual(len(samples), 6)

        expected_samples = []
        expected_samples.append(Node('a', [Node('b', [Node('c')])]))
        expected_samples.append(Node('a', [Node('c', [Node('b')])]))
        expected_samples.append(Node('b', [Node('a', [Node('c')])]))
        expected_samples.append(Node('b', [Node('c', [Node('a')])]))
        expected_samples.append(Node('c', [Node('a', [Node('b')])]))
        expected_samples.append(Node('c', [Node('b', [Node('a')])]))

        self.maxDiff = None
        self.assertCountEqual([sample
                               for _, sample in samples], expected_samples)

        self.assertEqual(idents, ['a', 'b', 'c'])
        self.assertEqual(classes, [0, 1, 2, 3, 4, 5])
