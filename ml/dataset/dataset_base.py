from torch.utils.data import Dataset

from common.utils import memoize


class DatasetBase(Dataset):

    def __init__(self, transform=None, preprocess=False):
        self.transform = transform
        self.preprocess = preprocess
        self._max_depth = -1
        self._max_spread = -1
        self.samples = []
        self._idents = []
        self.label_distribution = []
        self.patterns = []

    def unpack_sample(self, sample):
        raise NotImplementedError('unpack_sample')

    def _process_sample(self, sample):
        x, y = self.unpack_sample(sample)
        s = self._max_spread
        if self.transform is not None:
            return self.transform(x, y, s, spread=self._max_spread, depth=self._max_depth, idents=self._idents)
        return x, y, s

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
        return len(self._idents)

    @property
    def tag_size(self):
        # One additional for no tag
        return len(self.patterns) + 1

    @property
    def label_weight(self):
        min_node = min(self.label_distribution)
        return [min_node/max(label, 1) for label in self.label_distribution]
