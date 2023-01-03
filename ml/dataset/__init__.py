from pathlib import Path
from typing import Optional

import yaml


from .bag import BagDataset


class ScenarioParameter:
    def __init__(
        self,
        scenario: str,
        filename: str,
        solver_filename: str,
        data_size_limit: int,
        use_solver_data: bool,
        pattern_depth: Optional[int] = None,
        num_labels: Optional[int] = None,
        depth: Optional[int] = None,
        spread: Optional[int] = None,
        max_size: Optional[int] = None,
        **kwargs,
    ):
        self.scenario = scenario  # deprecated
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
    def from_config(config, use_solver_data=False):
        training = config.training
        files = config.files
        return ScenarioParameter(
            scenario=training.scenario,
            use_solver_data=use_solver_data,
            data_size_limit=training.data_size_limit,
            filename=files.trainings_data,
            solver_filename=files.solver_trainings_data,
        )

    @staticmethod
    def from_config_dict(config, use_solver_data=False):
        training = config['training']
        files = config['files']
        return ScenarioParameter(
            scenario=training['scenario'],
            use_solver_data=use_solver_data,
            data_size_limit=training['data-size-limit'],
            filename=files['trainings-data'],
            solver_filename=files['solver-trainings-data'],
        )

    @staticmethod
    def from_config_filename(filename: Path, use_solver_data=True):
        with filename.open() as f:
            return ScenarioParameter.from_config_dict(
                config=yaml.safe_load(f), use_solver_data=use_solver_data
            )

    @staticmethod
    def add_parsers(parser):
        parser.add_argument(
            '-s', '--scenario', type=str, default='pattern', choices=scenarios_choices()
        )
        parser.add_argument(
            '--depth', type=int, default=2, help='The depth of the used nodes.'
        )
        parser.add_argument(
            '--pattern-depth',
            type=int,
            default=1,
            help='The depth of the pattern nodes.',
        )
        parser.add_argument('--spread', type=int, default=2)
        parser.add_argument('--max-size', type=int, default=120)
        parser.add_argument('--num-labels', type=int, default=2)
        parser.add_argument('--bag-filename', type=str, default=None, dest='filename')
        parser.add_argument(
            '--solver-bag-filename', type=str, default=None, dest='solver_filename'
        )
        parser.add_argument('--use-solver-data', action='store_true', default=False)
        parser.add_argument(
            '--data-size-limit',
            type=int,
            default=None,
            help='Limits the size of the loaded bag file data. For testing purpose.',
        )


def scenarios_choices():
    return ['bag']


def create_scenario(params: ScenarioParameter, pad_token: int = 0, transform=None):
    if params.scenario == 'bag':
        return BagDataset.from_scenario_params(params=params, preprocess=True)
    else:
        raise NotImplementedError(f'Scenario {params.scenario} is not implemented!')
