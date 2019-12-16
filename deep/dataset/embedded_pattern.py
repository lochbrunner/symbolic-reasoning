from torch.utils.data import Dataset

from common.utils import memoize
from deep.dataset.generate import place_patterns_in_noise


class EmbPatternDataset(Dataset):
    '''Embeds different patterns in a noise'''

    def __init__(self, params, transform=None, preprocess=False):
        self.transform = transform
        self.samples, self.idents, self.patterns = place_patterns_in_noise(
            depth=params.depth, spread=params.spread, max_size=params.max_size, pattern_depth=1, num_labels=2)
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
        return len(self.patterns)
