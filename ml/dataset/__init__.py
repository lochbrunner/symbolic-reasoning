from common.utils import Compose

from .permutation import PermutationDataset
from .embedded_pattern import EmbPatternDataset
from .bag import BagDataset, BagDatasetSharedIndex
from .transformers import SegEmbedder, TagEmbedder, Padder, Uploader


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int, max_size: int,
                 pattern_depth: int, num_labels: int, filename: str, data_size_limit: int, **kwargs):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread
        self.max_size = max_size
        self.pattern_depth = pattern_depth
        self.num_labels = num_labels
        self.filename = filename
        self.data_size_limit = data_size_limit


def scenarios_choices():
    return ['permutation', 'pattern', 'bag', 'shared-bag']


def create_scenario(params: ScenarioParameter, device, pad_token=0, transform=None):
    if params.scenario == 'permutation':
        transform = transform or Compose([
            Padder(),
            TagEmbedder(),
            Uploader(device)
        ])
        return PermutationDataset(params=params, transform=transform)
    elif params.scenario == 'pattern':
        return EmbPatternDataset(params=params)
    elif params.scenario == 'shared-bag':
        return BagDatasetSharedIndex(params=params, preprocess=True)
    elif params.scenario == 'bag':
        return BagDataset(params=params, preprocess=True)
