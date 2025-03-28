import logging
import operator
from functools import reduce

import torch
import torch.optim as optim

from dataset import scenarios_choices, ScenarioParameter
from dataset.transformers import Embedder
from common.timer import Timer
from common.parameter_search import LearningParmeter

from .lstm_tagger import LstmTreeTagger, GruTreeTagger
from .fcn_tagger import FullyConnectedTagger
from .fcn_segmenter import FullyConnectedSegmenter
from .cnn_segmenter import TreeCnnSegmenter, TreeCnnUniqueIndices

all_models = {'LstmTreeTagger': LstmTreeTagger,
              'GruTreeTagger': GruTreeTagger,
              'FullyConnectedTagger': FullyConnectedTagger,
              'FullyConnectedSegmenter': FullyConnectedSegmenter,
              'TreeCnnSegmenter': TreeCnnSegmenter,
              'TreeCnnUniqueIndices': TreeCnnUniqueIndices
              }


def create_model(model_name, **kwargs):
    logging.info(f'Loading model {model_name}')
    if model_name in all_models:
        model = all_models[model_name](**kwargs)
    else:
        raise Exception(f'Unknown model {model_name}')

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')
    return model
