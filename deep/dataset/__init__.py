from .permutation import PermutationDataset
from .embedded_pattern import EmbPatternDataset


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread


def scenarios_choices():
    return ['permutation', 'pattern']
