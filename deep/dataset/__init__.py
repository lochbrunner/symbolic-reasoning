from .permutation import PermutationDataset
from .embedded_pattern import EmbPatternDataset

from .transformers import SegEmbedder, TagEmbedder, Padder, Uploader
from common.utils import Compose


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int, max_size: int):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread
        self.max_size = max_size


def scenarios_choices():
    return ['permutation', 'pattern']


def create_scenario(params: ScenarioParameter, device, transform=None, pad_token=0):
    if params.scenario == 'permutation':
        return PermutationDataset(params=params, transform=Compose([
            TagEmbedder(),
            Padder(pad_token=pad_token),
            Uploader(device)
        ]))
    elif params.scenario == 'pattern':
        return EmbPatternDataset(params=params, transform=Compose([
            SegEmbedder(),
            Padder(pad_token=pad_token),
            Uploader(device)
        ]))
