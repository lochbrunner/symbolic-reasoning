from torch.utils.data import Dataset
from common.utils import memoize

from deep.dataset.generate import create_samples_permutation


class PermutationDataset(Dataset):

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
