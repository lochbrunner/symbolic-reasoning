from common.utils import Compose

from .permutation import PermutationDataset
from .embedded_pattern import EmbPatternDataset
from .bag import BagDataset, BagDatasetSharedIndex
from .transformers import SegEmbedder, TagEmbedder, Padder, Uploader


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int, max_size: int,
                 pattern_depth: int, num_labels: int, filename: str, solver_filename: str,
                 data_size_limit: int, use_solver_data: bool, **kwargs):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread
        self.max_size = max_size
        self.pattern_depth = pattern_depth
        self.num_labels = num_labels
        self.data_size_limit = data_size_limit
        if use_solver_data:
            self.filename = solver_filename
        else:
            self.filename = filename

    @staticmethod
    def add_parsers(parser):
        parser.add_argument('-s', '--scenario', type=str,
                            default='pattern', choices=scenarios_choices())
        parser.add_argument('--depth', type=int, default=2,
                            help='The depth of the used nodes.')
        parser.add_argument('--pattern-depth', type=int, default=1,
                            help='The depth of the pattern nodes.')
        parser.add_argument('--spread', type=int, default=2)
        parser.add_argument('--max-size', type=int, default=120)
        parser.add_argument('--num-labels', type=int, default=2)
        parser.add_argument('--bag-filename', type=str, default=None, dest='filename')
        parser.add_argument('--solver-bag-filename', type=str, default=None, dest='solver_filename')
        parser.add_argument('--use-solver-data', action='store_true', default=False)
        parser.add_argument('--data-size-limit', type=int, default=None,
                            help='Limits the size of the loaded bag file data. For testing purpose.')


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
