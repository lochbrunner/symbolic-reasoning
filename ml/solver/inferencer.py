
import numpy as np
import torch

from pycore import Scenario

# project
from dataset.transformers import Padder, Embedder, ident_to_id
from models import create_model
from common import io
from common.timer import Timer
from common.parameter_search import LearningParmeter
from dataset import create_scenario
from dataset import ScenarioParameter


def get_model(exe_params, learn_params, scenario_params):
    # Creating dataset and model
    device = torch.device(exe_params.device)
    if exe_params.load_model:
        dataset, model, optimizer, _ = io.load(exe_params.exe_params.load_model, device)
    else:
        timer = Timer('Creating fresh workspace')
        dataset = create_scenario(params=scenario_params, device=device)
        model = create_model(learn_params.model_name,
                             hyper_parameter=learn_params.model_hyper_parameter,
                             **dataset.model_params)
        model.to(device)

        optimizer = torch.optim.Adadelta(model.parameters(), lr=learn_params.learning_rate)
        timer.stop_and_log()
    return dataset, model, optimizer


class Inferencer:
    ''' Standard inferencer for unique index map per sample
    '''

    def __init__(self, config, scenario: Scenario, fresh_model: bool):
        learn_params = LearningParmeter.from_config(config)
        if fresh_model:
            idents = scenario.idents
            self.spread = scenario.spread
            self.pad_token = 0  # Should be a constant
            self.model = create_model(learn_params.model_name,
                                      hyper_parameter=learn_params.model_hyper_parameter,
                                      vocab_size=scenario.vocab_size, tagset_size=scenario.tagset_size, pad_token=0, kernel_size=scenario.spread+2)
        else:
            self.model, snapshot = io.load_model(config.files.model)
            idents = snapshot['idents']
            self.spread = snapshot['kernel_size'] - 2
            self.pad_token = snapshot['pad_token']
            # self.weights = None

        self.model.eval()
        # Copy of BagDataset
        self.ident_dict = {ident: (value+1) for (value, ident) in enumerate(idents)}

    def __call__(self, initial, count=None):
        '''
        returns a tuple:
          * [(rule id, path)]
          * value
        '''

        # x, s, _ = self.dataset.embed_custom(initial)
        x, s, _, _, _ = initial.embed(self.ident_dict, self.pad_token, self.spread, [], True)
        x = torch.unsqueeze(torch.as_tensor(np.copy(x), device=self.model.device), 0)
        s = torch.unsqueeze(torch.as_tensor(np.copy(s), device=self.model.device), 0)
        p = torch.ones(x.shape[:-1])

        y, v = self.model(x, s, p)
        y = y.squeeze()  # shape: rules, localisation
        y = y.cpu().detach().numpy()[1:, :-1]  # Remove padding
        value = v.cpu().detach().numpy()[0][0]
        value = np.exp(value)

        parts_path = [p[0] for p in initial.parts_bfs_with_path]
        i = (-y).flatten().argsort()

        def calc(n):
            p = np.unravel_index(i[n], y.shape)
            return p[0]+1, parts_path[p[1]], y[p[0], p[1]]  # rule at path

        if count is None:
            count = i.shape[0]

        return [calc(i) for i in range(count)], value
