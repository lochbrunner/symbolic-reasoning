import numpy as np

import torch
from torch.utils.data import Dataset

from pycore import Bag

from .transformers import PAD_INDEX
from .symbol_builder import SymbolBuilder
from .dataset_base import DatasetBase
from common.node import Node
from common.terminal_utils import printProgressBar, clearProgressBar

import logging


class BagDatasetSharedIndex(DatasetBase):
    '''Deprecated!

    Loads samples from a bag file'''

    def __init__(self, params, preprocess=False):
        super(BagDatasetSharedIndex, self).__init__(preprocess)
        logging.info(f'Loading samples from {params.filename}')
        bag = Bag.load(params.filename)

        self.patterns = [rule.condition for rule in bag.meta.rules]

        meta = bag.meta

        self.idents = meta.idents
        self.label_distribution = meta.rule_distribution
        self._rule_map = list(meta.rules)

        # Merge use largest
        self.container = [sample for container in bag.samples for sample in container.samples]
        self._max_spread = bag.samples[-1].max_spread
        self._max_depth = bag.samples[-1].max_depth

        def create_features(c):
            return [(c.initial, fit) for fit in c.fits]

        self.raw_samples = [feature for sample in self.container
                            for feature in create_features(sample)]

        logging.info(f'number of samples: {len(self.raw_samples)}')
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

    @property
    def model_params(self):
        return {
            'vocab_size': len(self.idents),
            'tagset_size': len(self.patterns),
            'pad_token': PAD_INDEX,
            'kernel_size': self._max_spread+2,
            'depth': self.max_depth,
            'spread': self._max_spread,
            'idents': self.idents
        }


class BagDataset(Dataset):

    pad_token = 0
    spread = 2

    def __init__(self, params, preprocess=False):
        self.preprocess = preprocess

        logging.info(f'Loading samples from {params.filename}')
        bag = Bag.load(params.filename)

        meta = bag.meta

        self.rule_conditions = [rule.condition for rule in meta.rules]

        self.idents = meta.idents

        # 0 is padding
        self._ident_dict = {ident: (value+1) for (value, ident) in enumerate(self.idents)}

        self.label_distribution = meta.rule_distribution
        self._rule_map = list(meta.rules)

        # Merge use largest
        self.container = [sample for container in bag.samples for sample in container.samples]
        self._max_spread = bag.samples[-1].max_spread
        self._max_depth = bag.samples[-1].max_depth
        self._max_size = bag.samples[-1].max_size
        logging.info(f'max size: {self._max_size}')
        logging.info(f'number of samples: {len(self.container)}')

        if preprocess:
            def progress(i, sample):
                if i % 50 == 0:
                    printProgressBar(i, len(self.container), suffix='loading')
                return sample
            self.samples = [progress(i, self._process_sample(sample)) for i, sample in enumerate(self.container)]
            clearProgressBar()
        else:
            self.samples = self.container

    # def get_node(self, index):
    #     return Node.from_rust(self.raw_samples[index][0])

    # def get_rule_of_sample(self, index):
    #     rule_id = self.raw_samples[index][1].rule
    #     return self.get_rule_raw(rule_id)

    # def get_node_string(self, index):
    #     return str(self.raw_samples[index][0])

    def _process_sample(self, sample):
        # x, fit = sample
        x, s, y = sample.initial.embed(self._ident_dict, self.pad_token, self.spread, sample.fits)
        padding_index = x.shape[0] - 1

        # TODO: Do padding in collate function
        # such that we do not need global max size
        pad_width = self._max_size - x.shape[0]
        x = np.pad(x, (0, pad_width), 'constant', constant_values=(self.pad_token))
        pad_width = self._max_size - s.shape[0]
        s = np.pad(s, ((0, pad_width), (0, 0)), 'constant', constant_values=(padding_index))
        pad_width = self._max_size - y.shape[0]
        y = np.pad(y, (0, pad_width), 'constant', constant_values=(self.pad_token))

        x = torch.as_tensor(x)
        s = torch.as_tensor(s)
        y = torch.as_tensor(y)
        return x, s, y

    @property
    def max_depth(self):
        return self._max_depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if not self.preprocess:
            return self._process_sample(self.samples[index])
        else:
            return self.samples[index]

    @property
    def tag_size(self):
        # One additional for no tag
        return len(self.rule_conditions)

    @property
    def rule_map(self):
        '''Maps rule id to rule string representation'''
        return self._rule_map

    def get_rule_raw(self, index):
        return self._rule_map[index]

    def get_rules_raw(self):
        return self._rule_map

    @property
    def label_weight(self):
        min_node = max(min(self.label_distribution), 1)
        return [min_node/max(label, 1) for label in self.label_distribution]

    @property
    def model_params(self):
        return {
            'vocab_size': len(self.idents),
            'tagset_size': len(self.rule_conditions),
            'pad_token': self.pad_token,
            'kernel_size': self.spread+2,
            'idents': self.idents
        }
