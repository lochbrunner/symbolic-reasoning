from pycore import Bag


from .symbol_builder import SymbolBuilder
from .dataset_base import DatasetBase
from node import Node


class BagDataset(DatasetBase):
    '''Loads samples from a bag file'''

    def __init__(self, params, preprocess=False):
        super(BagDataset, self).__init__(preprocess)

        bag = Bag.load(params.filename)

        self.patterns = [rule.condition for rule in bag.meta.rules]

        meta = bag.meta

        self._idents = meta.idents
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

        builder = SymbolBuilder()
        for _ in range(self._max_depth):
            builder.add_level_uniform(self._max_spread)
        self.label_builder = builder

        if preprocess:
            self.samples = [self._process_sample(sample) for sample in self.raw_samples]

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
