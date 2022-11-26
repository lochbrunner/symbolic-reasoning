from typing import Optional, Sequence
import abc
import numpy as np
import torch

# project
from models import create_model
from common import io
from common.timer import Timer
from common.parameter_search import LearningParameter
from dataset import create_scenario
from dataset import ScenarioParameter

from pycore import Scenario, Symbol, Rule, fit

Path = Sequence[int]


def get_model(exe_params, learn_params, scenario_params: ScenarioParameter):
    # Creating dataset and model
    device = torch.device(exe_params.device)
    if exe_params.load_model:
        dataset, model, optimizer, _ = io.load(exe_params.exe_params.load_model, device)
    else:
        timer = Timer('Creating fresh workspace')
        dataset = create_scenario(params=scenario_params)
        model = create_model(
            learn_params.model_name,
            hyper_parameter=learn_params.model_hyper_parameter,
            **dataset.model_params,
        )
        model.to(device)

        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=learn_params.learning_rate
        )
        timer.stop_and_log()
    return dataset, model, optimizer


class Inferencer(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, initial: Symbol, count: Optional[int] = None
    ) -> tuple[Sequence[tuple[int, Path, float]], float]:
        pass


def _policy_field_to_fitinfo(
    y: np.ndarray,
    count: Optional[int],
    initial: Symbol,
    rule_offset: int = 1,
) -> Sequence[tuple[int, Path, float]]:

    parts_path = [p[0] for p in initial.parts_bfs_with_path]
    i = (-y).flatten().argsort()

    def calc(n) -> tuple[int, Path, float]:
        [rule_id_np, path_id_np, *_] = np.unravel_index(i[n], y.shape)
        rule_id = int(rule_id_np)
        path_id = int(path_id_np)
        return (
            rule_id + rule_offset,
            parts_path[path_id],
            y[rule_id, path_id],
        )  # rule at path and confidence

    return [calc(i) for i in range(count or i.shape[0])]


class SophisticatedInferencer(Inferencer):
    def __init__(self, scenario: Scenario, rule_mapping: dict[int, Rule]) -> None:
        super().__init__()
        self._tagset_size = scenario.tagset_size
        self._rng = np.random.default_rng(0)
        self._rule_mapping = rule_mapping

    def __call__(
        self, initial: Symbol, count: Optional[int] = None
    ) -> tuple[Sequence[tuple[int, Path, float]], float]:
        fits = []
        for rule_id, rule in self._rule_mapping.items():
            for fit_map in fit(initial, rule.condition):
                fits.append((rule_id, fit_map.path, 1.0))
                if count is not None and count == len(fits):
                    return fits, 1.0

        return fits, 1.0


class RandomInferencer(Inferencer):
    def __init__(self, scenario: Scenario) -> None:
        super().__init__()
        self._tagset_size = scenario.tagset_size
        self._rng = np.random.default_rng(0)

    def __call__(
        self, initial: Symbol, count: Optional[int] = None
    ) -> tuple[Sequence[tuple[int, Path, float]], float]:
        parts_path = [p[0] for p in initial.parts_bfs_with_path]
        y = self._rng.uniform(
            low=-1, high=1.0, size=(self._tagset_size - 1, len(parts_path))
        )
        value = self._rng.uniform(
            low=-1,
            high=1.0,
        )

        return _policy_field_to_fitinfo(y, count, initial, rule_offset=2), value


class TorchInferencer(Inferencer):
    '''Standard inferencer for unique index map per sample'''

    def __init__(self, config, scenario: Scenario, fresh_model: bool):
        learn_params = LearningParameter.from_config(config)
        if fresh_model:
            idents = scenario.idents()
            self.spread = scenario.spread
            self.pad_token = 0  # Should be a constant
            self.model = create_model(
                learn_params.model_name,
                hyper_parameter=learn_params.model_hyper_parameter,
                vocab_size=scenario.vocab_size(),
                tagset_size=scenario.tagset_size,
                pad_token=0,
                kernel_size=scenario.spread + 2,
            )
            self.trained_metrics = None
        else:
            self.model, snapshot = io.load_model(config.files.model)
            idents = snapshot['idents']
            self.spread = snapshot['kernel_size'] - 2
            self.pad_token = snapshot['pad_token']
            self.trained_metrics = snapshot.get('metrics', None)
            # self.weights = None

        self.model.eval()
        # Copy of BagDataset
        self.target_size = scenario.tagset_size
        self.ident_dict = {ident: (value + 1) for (value, ident) in enumerate(idents)}

    def inference(self, initial: Symbol, keep_padding=False):
        x, s, _, _, _, _, _, _ = initial.embed(
            self.ident_dict,
            self.pad_token,
            self.spread,
            initial.depth,
            self.target_size,
            [],
            True,
            index_map=True,
            positional_encoding=False,
        )
        x = torch.unsqueeze(torch.as_tensor(np.copy(x), device=self.model.device), 0)
        s = torch.unsqueeze(torch.as_tensor(np.copy(s), device=self.model.device), 0)
        p = torch.ones(x.shape[:-1])
        y, v = self.model(x, s, p)
        y = y.squeeze()  # shape: rules, path
        y = y.cpu().detach().numpy()
        if not keep_padding:
            y = y[1:, :-1]  # Remove padding
        value = v.cpu().detach().numpy()[0][0]
        value = np.exp(value)
        return y, value

    def __call__(self, initial: Symbol, count: Optional[int] = None):
        '''
        returns a tuple:
          * [(rule id, path, confidence)]
          * value
        '''

        # x, s, _ = self.dataset.embed_custom(initial)
        # x, s, _, _, _, _ = initial.embed(self.ident_dict, self.pad_token, self.spread,
        #                                  initial.depth, [], True, index_map=True, positional_encoding=False)
        # x = torch.unsqueeze(torch.as_tensor(np.copy(x), device=self.model.device), 0)
        # s = torch.unsqueeze(torch.as_tensor(np.copy(s), device=self.model.device), 0)
        # p = torch.ones(x.shape[:-1])
        # y, v = self.model(x, s, p)
        # y = y.squeeze()  # shape: rules, path
        # y = y.cpu().detach().numpy()[1:, :-1]  # Remove padding
        # value = v.cpu().detach().numpy()[0][0]
        # value = np.exp(value)
        y, value = self.inference(initial)
        return _policy_field_to_fitinfo(y, count, initial), value

        # parts_path = [p[0] for p in initial.parts_bfs_with_path]
        # i = (-y).flatten().argsort()

        # def calc(n):
        #     p = np.unravel_index(i[n], y.shape)
        #     return (
        #         p[0] + 1,
        #         parts_path[p[1]],
        #         y[p[0], p[1]],
        #     )  # rule at path and confidence

        # return [calc(i) for i in range(count or i.shape[0])], value
