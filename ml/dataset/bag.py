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

logger = logging.getLogger(__name__)


class BagDatasetSharedIndex(DatasetBase):
    '''Deprecated!

    Loads samples from a bag file'''

    def __init__(self, params, preprocess=False):
        super(BagDatasetSharedIndex, self).__init__(preprocess)
        logger.info(f'Loading samples from {params.filename}')
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

        logger.info(f'number of samples: {len(self.raw_samples)}')
        logger.info(f'max depth: {self._max_depth}')

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

    @property
    def collate_fn(self):
        return None


def pad(sample, width, pad_token=0):
    '''Pad in first dimension'''
    pad_width = width - sample.shape[0]
    if sample.ndim == 1:
        return np.pad(sample, (0, pad_width), 'constant', constant_values=(pad_token))
    elif sample.ndim == 2:
        return np.pad(sample, ((0, pad_width), (0, 0)), 'constant', constant_values=(pad_token))
    else:
        raise NotImplementedError(f'Padding of {sample.ndim} dim tensor not implemented yet!')


def stack(samples, width):

    samples = [torch.as_tensor(pad(sample, width)) for sample in samples]

    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        elem = samples[0]
        numel = sum([x.numel() for x in samples])
        storage = elem.storage()._new_shared(numel)  # pylint: disable=protected-access
        out = elem.new(storage)
    return torch.stack(samples, 0, out=out)


def dynamic_width_collate(batch):
    max_width = max([sample[0].shape[0] for sample in batch])
    # Transpose them
    transposed = zip(*batch)

    return [stack(channel, max_width) for channel in transposed]


class BagDataset(Dataset):

    pad_token = 0
    spread = 2

    def __init__(self, params, preprocess=False):
        self.preprocess = preprocess

        logger.info(f'Loading samples from {params.filename}')
        bag = Bag.load(params.filename)

        meta = bag.meta

        self.rule_conditions = [rule.condition for rule in meta.rules]

        self.idents = meta.idents

        # 0 is padding
        self._ident_dict = {ident: (value+1) for (value, ident) in enumerate(self.idents)}

        self.label_distribution = meta.rule_distribution
        self._rule_map = list(meta.rules)

        # Merge use largest

        if params.data_size_limit is None:
            limit = -1
        else:
            limit = params.data_size_limit

        self.container = [sample for container in bag.samples for sample in container.samples][:limit]
        self._max_spread = bag.samples[-1].max_spread
        self._max_depth = bag.samples[-1].max_depth
        self._max_size = bag.samples[-1].max_size
        logger.info(f'max size: {self._max_size}')
        logger.info(f'number of samples: {len(self.container)}')

        if preprocess:
            def progress(i, sample):
                if i % 50 == 0:
                    printProgressBar(i, len(self.container), suffix='loading')
                return sample
            self.samples = [progress(i, self._process_sample(sample)) for i, sample in enumerate(self.container)]
            clearProgressBar()
        else:
            self.samples = self.container

    def get_node(self, index):
        return Node.from_rust(self.container[index].initial)

    def get_sample(self, index):
        return dynamic_width_collate([self[index]])

    def get_rule_of_sample(self, index):
        rule_id = self.container[index].fits[0].rule
        return self._rule_map[rule_id]

    def _process_sample(self, sample):
        return sample.initial.embed(self._ident_dict, self.pad_token, self.spread, sample.fits)

    def embed_custom(self, initial, fits=None):
        return initial.embed(self._ident_dict, self.pad_token, self.spread, fits or [])

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

    @property
    def collate_fn(self):
        return dynamic_width_collate
