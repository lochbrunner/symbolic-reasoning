from pycore import Bag


from .symbol_builder import SymbolBuilder
from .dataset_base import DatasetBase
from common.node import Node
from common.terminal_utils import printProgressBar, clearProgressBar

import logging


class BagDataset(DatasetBase):
    '''Loads samples from a bag file'''

    def __init__(self, params, preprocess=False):
        super(BagDataset, self).__init__(preprocess)
        bag = Bag.load(params.filename)

        self.patterns = [rule.condition for rule in bag.meta.rules]

        meta = bag.meta

        self.idents = meta.idents
        self.label_distribution = meta.rule_distribution
        self._rule_map = [rule for rule in meta.rules]

        # Only use largest
        self.container = bag.samples[-1]
        self._max_spread = self.container.max_spread
        self._max_depth = self.container.max_depth

        def create_features(c):
            return [(c.initial, fit) for fit in c.fits]

        self.raw_samples = [feature for sample in self.container.samples
                            for feature in create_features(sample)]

        logging.info(f'#samples: {len(self.raw_samples)}')
        logging.info(f'max depth: {self._max_depth}')

        builder = SymbolBuilder()
        for _ in range(self._max_depth):
            builder.add_level_uniform(self._max_spread)
        self.label_builder = builder

        if preprocess:
            def progress(i, sample):
                if i % 50 == 0:
                    printProgressBar(i, len(self.raw_samples), suffix='loading')
                return sample
            self.samples = [progress(i, self._process_sample(sample)) for i, sample in enumerate(self.raw_samples)]
            clearProgressBar()
        else:
            self.samples = self.raw_samples

    def get_node(self, index):
        return Node.from_rust(self.raw_samples[index][0])

    def get_rule_of_sample(self, index):
        rule_id = self.raw_samples[index][1].rule
        return self.get_rule_raw(rule_id)

    def get_node_string(self, index):
        return str(self.raw_samples[index][0])

    def unpack_sample(self, sample):
        x, fit = sample
        return x, (fit.path, fit.rule)

    @property
    def rule_map(self):
        '''Maps rule id to rule string representation'''
        return self._rule_map

    def get_rule_raw(self, index):
        return self._rule_map[index]

    def get_rules_raw(self):
        return self._rule_map
