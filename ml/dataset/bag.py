import numpy as np

import torch
from torch.utils.data import Dataset

from pycore import Bag, BagMeta

from common.node import Node
from common.terminal_utils import printProgressBar, clearProgressBar

import logging

logger = logging.getLogger(__name__)


def pad(sample, width, pad_token=0):
    '''Pad in first dimension'''
    if sample is None:
        return None
    pad_width = width - sample.shape[0]
    if sample.ndim == 1:
        return np.pad(sample, (0, pad_width), 'constant', constant_values=(pad_token))
    elif sample.ndim == 2:
        return np.pad(sample, ((0, pad_width), (0, 0)), 'constant', constant_values=(pad_token))
    else:
        raise NotImplementedError(f'Padding of {sample.ndim} dim tensor not implemented yet!')


def stack(samples, width=None):
    if width is None:
        samples = [torch.as_tensor(sample) for sample in samples if sample is not None]
    else:
        samples = [torch.as_tensor(pad(sample, width)) for sample in samples if sample is not None]

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
    # x, s?, o?, y, p, v
    max_width = max(sample[0].shape[0] for sample in batch)
    # Transpose them
    transposed = list(zip(*batch))
    # Don't pad value
    return [stack(channel, max_width) for channel in transposed[:-1]] + [stack(channel) for channel in transposed[-1:]]


class BagDataset(Dataset):
    '''
    > Note: Returning numpy arrays no torch tensors. 
    '''

    pad_token = 0
    spread = 2

    @staticmethod
    def from_scenario_params(params, preprocess=False):
        return BagDataset.load(filename=params.filename, data_size_limit=params.data_size_limit, preprocess=preprocess)

    @staticmethod
    def from_container(container, meta: BagMeta, data_size_limit: int = None, preprocess: bool = False):
        return BagDataset(meta, container.samples, container.max_depth, container.max_size, data_size_limit, preprocess)

    @staticmethod
    def load(filename, data_size_limit=None, preprocess=False):
        logger.info(f'Loading samples from {filename}')
        bag = Bag.load(str(filename))
        samples = [sample for container in bag.containers for sample in container.samples]
        max_depth = bag.containers[-1].max_depth
        max_size = bag.containers[-1].max_size
        return BagDataset(bag.meta, samples, max_depth, max_size, data_size_limit, preprocess)

    def __init__(self, meta, samples, max_depth, max_size, data_size_limit=None, preprocess=False):
        self.preprocess = preprocess

        self.rule_conditions = [rule.condition for rule in meta.rules]

        self.idents = meta.idents

        # 0 is padding
        self._ident_dict = {ident: (value+1) for (value, ident) in enumerate(self.idents)}

        self.label_distribution = [p+n for p, n in meta.rule_distribution]
        self.value_distribution = meta.value_distribution
        self._rule_map = meta.rules

        # Merge use largest

        if data_size_limit is None:
            limit = -1
        else:
            limit = data_size_limit

        self.container = samples[:limit]
        self._max_depth = max_depth
        self._max_size = max_size
        logger.debug(f'max size: {self._max_size}')
        logger.debug(f'number of samples: {len(self.container)}')
        logger.debug(f'number of rules: {len(self._rule_map)}')

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
        channels = sample.embed(self._ident_dict, self.pad_token, self.spread,
                                self._max_depth, index_map=True, positional_encoding=False)
        return [c for c in channels if c is not None]

    def embed_custom(self, initial, fits=None, useful=True):
        return initial.embed(self._ident_dict, self.pad_token, self.spread, self._max_depth, fits or [], useful, index_map=True, positional_encoding=False)

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
        return np.array([min_node/max(label, 1) for label in self.label_distribution], dtype=np.float32)

    @property
    def value_weight(self):
        p, n = self.value_distribution
        min_dist = max(min(self.value_distribution), 1)
        return np.array([p, n], np.float32) / min_dist

    @property
    def model_params(self):
        return {
            'vocab_size': len(self.idents),
            'tagset_size': len(self.rule_conditions),
            'pad_token': self.pad_token,
            'kernel_size': self.spread+2,
            'idents': self.idents
        }

    collate_fn = dynamic_width_collate

    def get_collate_fn(self):
        return dynamic_width_collate
