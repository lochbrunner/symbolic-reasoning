import logging
import operator
from functools import reduce

import torch
import torch.optim as optim

from dataset import scenarios_choices, ScenarioParameter
from common.timer import Timer
from common.parameter_search import LearningParmeter

from .cnn_segmenter import TreeCnnSegmenter

logger = logging.getLogger(__name__)

all_models = {
    'TreeCnnSegmenter': TreeCnnSegmenter,
}


def create_model(model_name, **kwargs):
    logger.info(f'Creating model {model_name}')
    if model_name in all_models:
        model = all_models[model_name](**kwargs)
    else:
        raise Exception(f'Unknown model {model_name}')

    num_parameters = sum([reduce(
        operator.mul, p.size()) for p in model.parameters()])
    logger.info(f'Number of parameters: {num_parameters}')
    return model
