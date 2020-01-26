import torch
from torch.utils.data import Dataset

from common.utils import memoize
from common.node import Node
from .symbol_builder import SymbolBuilder
from .transformers import Padder, Embedder, ident_to_id


class DatasetBase(Dataset):

    def __init__(self, preprocess=False):
        self.preprocess = preprocess
        self._max_depth = -1
        self._max_spread = -1
        self.samples = []
        self.idents = []
        self.label_distribution = []
        self.patterns = []

    def unpack_sample(self, sample):
        raise NotImplementedError('unpack_sample')

    def _process_sample(self, sample):
        # pad
        x, (path, label) = self.unpack_sample(sample)
        builder = Padder.create_mask(x)
        builder.set_label_at(path, label)
        y = builder.symbol_ref
        x = Padder.pad(x, spread=self._max_spread, depth=self._max_depth)

        def factory():
            return Node(label=-1, childs=[])
        y = Padder.pad(y, factory=factory, spread=self._max_spread, depth=self._max_depth)

        # unroll
        x = [ident_to_id(n, self.idents) for n in Embedder.unroll(x)] + [0]
        y = [n.label or 0 for n in Embedder.unroll(y)] + [-1]
        # torchify
        x = torch.as_tensor(x, dtype=torch.long)
        y = torch.as_tensor(y, dtype=torch.long)
        return x, y

    @property
    def max_spread(self):
        return self._max_spread

    @property
    def max_depth(self):
        return self._max_depth

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
        return len(self.patterns)

    @property
    def label_weight(self):
        min_node = max(min(self.label_distribution), 1)
        return [min_node/max(label, 1) for label in self.label_distribution]
