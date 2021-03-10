import logging
import operator
from pathlib import Path
from functools import reduce

import torch
import torch.optim as optim

from dataset import create_scenario
from models import create_model
from .timer import Timer


def load(filename, device=torch.device('cpu'), transform=None):
    with Timer(f'Loading snapshot from {filename}'):
        snapshot = torch.load(filename)
        scenario_params = snapshot['scenario_parameter']
        padding_index = 0

        dataset = create_scenario(params=scenario_params, device=device,
                                  pad_token=padding_index, transform=transform)

        learn_params = snapshot['learning_parameter']
        model = create_model(learn_params.model_name,
                             **dataset.model_params,
                             hyper_parameter=learn_params.model_hyper_parameter)

        model.load_state_dict(snapshot['model_state_dict'])
        optimizer = optim.SGD(model.parameters(), lr=learn_params.learning_rate)

    return dataset, model, optimizer, scenario_params


def load_model(filename, spread=None, depth=None, kernel_size=None):
    with Timer(f'Loading model from snapshot {filename}'):
        snapshot = torch.load(filename)
        learn_params = snapshot['learning_parameter']
        padding_index = 0
        tag_size = snapshot['tagset_size']
        vocab_size = snapshot['vocab_size']
        model = create_model(learn_params.model_name,
                             vocab_size=vocab_size,
                             tagset_size=tag_size,
                             pad_token=padding_index,
                             spread=spread or snapshot['spread'] if 'spread' in snapshot else None,
                             depth=depth or snapshot['depth'] if 'depth' in snapshot else None,
                             kernel_size=kernel_size or snapshot['kernel_size'] if 'kernel_size' in snapshot else None,
                             hyper_parameter=learn_params.model_hyper_parameter)
        model.load_state_dict(snapshot['model_state_dict'])
        return model, snapshot


def save(filename, model, optimizer, scenario_params, learn_params, dataset, metrics=None):
    if filename is None:
        return
    filename = Path(filename)
    logging.info(f'Writing snapshot to {filename}')
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'model_name': model.__class__.__name__,
             'learning_parameter': learn_params,
             'scenario_parameter': scenario_params,
             'rules': [rule.verbose for rule in dataset.get_rules_raw()],  # For consistency checks
             'metrics': metrics
             }
    state.update(dataset.model_params)
    filename.parent.mkdir(exist_ok=True)
    torch.save(state, filename)
