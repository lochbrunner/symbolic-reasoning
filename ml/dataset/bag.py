from torch.utils.data import Dataset

from pycore import Bag

from common.utils import memoize
from .symbol_builder import SymbolBuilder
from .dataset_base import DatasetBase


class BagDataset(DatasetBase):
    '''Loads samples from a bag file'''

    def __init__(self, params, transform=None, preprocess=False):
        super(BagDataset, self).__init__(transform, preprocess)

        bag = Bag.load(params.filename)

        self.patterns = [rule.condition for rule in bag.meta.rules]

        meta = bag.meta

        self._idents = meta.idents
        self.label_distribution = meta.rule_distribution

        # Only use largest
        container = bag.samples[-1]
        self._max_spread = container.max_spread
        self._max_depth = container.max_depth

        def create_features(c):
            return [(c.initial, fit) for fit in c.fits]

        self.samples = [feature for sample in container.samples
                        for feature in create_features(sample)]

        builder = SymbolBuilder()
        for _ in range(self._max_depth):
            builder.add_level_uniform(self._max_spread)
        self.label_builder = builder

        if preprocess:
            self.samples = [self._process_sample(sample) for sample in self.samples]

    def unpack_sample(self, sample):
        x, fit = sample

        builder = self.label_builder
        builder.clear_labels()
        # Label 0 means no label
        builder.set_label_at(fit.path, fit.rule)
        y = builder.symbol
        return x, y
