import logging
import operator
from functools import reduce

import torch
import torch.optim as optim

from deep.dataset import PermutationDataset, Embedder, Padder, Uploader, scenarios_choices, ScenarioParameter
from common.timer import Timer
from common.parameter_search import LearningParmeter

from .lstm_tree_tagger import LstmTreeTagger, GruTreeTagger
from .fcn_tagger import FullyConnectedTagger

all_models = {'LstmTreeTagger': LstmTreeTagger,
              'GruTreeTagger': GruTreeTagger,
              'FullyConnectedTagger': FullyConnectedTagger}


def create_model(model_name, vocab_size, tagset_size, pad_token, blueprint, hyper_parameter):
    if model_name in all_models:
        return all_models[model_name](vocab_size, tagset_size, pad_token, blueprint, hyper_parameter)
    else:
        raise Exception(f'Unknown model {model_name}')


def load_model(filename, dataset, learn_params: LearningParmeter, scenario_params: ScenarioParameter, pad_token):
    if learn_params.model_name is None and filename is not None:
        model_name = checkpoint = torch.load(filename)['model_name']
    else:
        model_name = learn_params.model_name
    model = create_model(model_name,
                         vocab_size=dataset.vocab_size,
                         tagset_size=dataset.tag_size,
                         pad_token=pad_token,
                         blueprint=Embedder.blueprint(scenario_params),
                         hyper_parameter=learn_params.model_hyper_parameter)

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    optimizer = optim.SGD(model.parameters(), lr=learn_params.learning_rate)

    if filename is not None:
        # TODO: Find suitable model with the given hyper parameters
        timer = Timer(f'Loading model from {filename}')
        checkpoint = torch.load(filename)
        checkpoint_model_name = checkpoint['model_name']
        if checkpoint_model_name != model.__class__.__name__:
            raise Exception(
                f'The model in file {filename} is of type {checkpoint_model_name} but expected {model.__class__.__name__}')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        timer.stop_and_log()

    return model, optimizer


def save_model(filename, model, optimizer, iteration, learn_params: LearningParmeter):
    if filename is None:
        return
    logging.info(f'Saving model to {filename} ...')
    torch.save({
        'epoch': learn_params.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'model_name': model.__class__.__name__,
        'hyper_parameter': learn_params.model_hyper_parameter}, filename)
