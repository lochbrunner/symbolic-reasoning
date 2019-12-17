import logging
import operator
from functools import reduce

import torch
import torch.optim as optim

from deep.dataset import create_scenario
from deep.models import create_model
from deep.dataset.transformers import Embedder
from common.timer import Timer


def load(filename, device=torch.device('cpu'), transform=None):
    timer = Timer(f'Loading snapshot from {filename}')
    snapshot = torch.load(filename)
    scenario_params = snapshot['scenario_parameter']
    padding_index = 0

    dataset = create_scenario(params=scenario_params, device=device,
                              pad_token=padding_index, transform=transform)

    learn_params = snapshot['learning_parameter']
    model = create_model(learn_params.model_name,
                         vocab_size=dataset.vocab_size,
                         tagset_size=dataset.tag_size,
                         pad_token=padding_index,
                         blueprint=Embedder.blueprint(scenario_params),
                         hyper_parameter=learn_params.model_hyper_parameter)

    optimizer = optim.SGD(model.parameters(), lr=learn_params.learning_rate)
    timer.stop_and_log()
    return dataset, model, optimizer, scenario_params


def save(filename, model, optimizer, scenario_params, learn_params):
    logging.info(f'Writing snapshot to {filename}')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': model.__class__.__name__,
        'learning_parameter': learn_params,
        'scenario_parameter': scenario_params,
    }, filename)
