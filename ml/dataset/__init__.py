from common.utils import Compose

from .permutation import PermutationDataset
from .embedded_pattern import EmbPatternDataset
from .bag import BagDataset
from .transformers import SegEmbedder, TagEmbedder, Padder, Uploader


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int, max_size: int,
                 pattern_depth: int, num_labels: int, filename: str, **kwargs):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread
        self.max_size = max_size
        self.pattern_depth = pattern_depth
        self.num_labels = num_labels
        self.filename = filename


def scenarios_choices():
    return ['permutation', 'pattern', 'bag']


def create_scenario(params: ScenarioParameter, device, pad_token=0, transform=None):
    if params.scenario == 'permutation':
        transform = transform or Compose([
            Padder(),
            TagEmbedder(),
            Uploader(device)
        ])
        return PermutationDataset(params=params, transform=transform)
    elif params.scenario == 'pattern':
        transform = transform or Compose([
            Padder(),
            SegEmbedder(),
            Uploader(device)
        ])
        return EmbPatternDataset(params=params, transform=transform)
    elif params.scenario == 'bag':
        transform = transform or Compose([
            Padder(),
            SegEmbedder(),
            Uploader(device)
        ])
        return BagDataset(params=params, transform=transform, preprocess=True)
