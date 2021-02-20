from typing import List
from copy import deepcopy
from argparse import Namespace


class LearningParmeter:
    def __init__(self, model_name, num_epochs: int = 10, learning_rate: float = 0.1,
                 batch_size: int = 10, gradient_clipping: float = 0.1,
                 value_loss_weight: float = 0.5,
                 model_hyper_parameter: dict = None,
                 optimizer: str = "adadelta",
                 fine_tuning=None, **kwargs):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.value_loss_weight = value_loss_weight
        self.optimizer = optimizer
        if isinstance(model_hyper_parameter, Namespace):
            self.model_hyper_parameter = vars(model_hyper_parameter)
        else:
            self.model_hyper_parameter = model_hyper_parameter or {}

        self.fine_tuning = fine_tuning or {}

    def use_finetuning(self):
        for k, v in self.fine_tuning.items():
            if not hasattr(self, k):
                raise RuntimeError(f'Learning parameter {k} not found!')
            setattr(self, k, v)

    @staticmethod
    def from_config(config):
        training = config.training
        return LearningParmeter(model_name=training.model_name,
                                num_epochs=training.num_epochs,
                                learning_rate=training.learning_rate,
                                batch_size=training.batch_size,
                                gradient_clipping=training.gradient_clipping,
                                value_loss_weight=training.value_loss_weight,
                                model_hyper_parameter=vars(training.model_parameter),
                                optimizer=training.optimizer,
                                fine_tuning=vars(config.fine_tuning)
                                )

    @staticmethod
    def from_config_dict(config):
        training = config['training']
        return LearningParmeter(model_name=training['model-name'],
                                num_epochs=training['num-epochs'],
                                learning_rate=training['learning-rate'],
                                batch_size=training['batch-size'],
                                gradient_clipping=training['gradient-clipping'],
                                value_loss_weight=training['value-loss-weight'],
                                model_hyper_parameter=training['model-parameter'],
                                fine_tuning=config['fine-tuning']
                                )

    @staticmethod
    def from_config_and_hyper(config, model_hyper_parameter):
        training = config.training
        return LearningParmeter(model_name=training.model_name,
                                num_epochs=training.num_epochs,
                                learning_rate=training.learning_rate,
                                batch_size=training.batch_size,
                                gradient_clipping=training.gradient_clipping,
                                value_loss_weight=training.value_loss_weight,
                                model_hyper_parameter=model_hyper_parameter,
                                fine_tuning=vars(config.fine_tuning)
                                )

    @staticmethod
    def add_parsers(parser, all_models: list):
        parser.add_argument('-n', '--num-epochs', type=int, default=30)
        parser.add_argument('-b', '--batch-size', type=int, default=32)
        parser.add_argument('-l', '--learning-rate', type=float, default=1.0)
        parser.add_argument('-g', '--gradient-clipping', type=float, default=0.1)
        parser.add_argument('-m', '--model', choices=all_models,
                            default='TreeCnnSegmenter', dest='model_name')
        parser.add_argument('--value-loss-weight', type=float, default=0.5)


class MarchSearch:
    '''
    Assumes that the parameter are roughly independent.

    Evaluation policy (finding new direction)
    * begin
    * increasing step
    * a) fixed period
    * b) when things went worse
    '''

    def __init__(self, init: LearningParmeter = LearningParmeter('LstmTreeTagger')):
        self.param = {
            'common': init.__dict__,
            'model': init.model_hyper_parameter
        }
        self.state = 'init'
        self.param_names = [
            ('common', k) for k in self.param['common'] if k != 'model_hyper_parameter']

        self.param_names += [('model', k) for k in self.param['model']]
        self.current_param_index = 0
        self.last_loss: float = None

        self.direction = {'common': {}, 'model': {}}
        self.velocity = 1.

    def _get_current(self):
        param = LearningParmeter('LstmTreeTagger')
        param.__dict__ = self.param['common']
        param.model_hyper_parameter = self.param['model']
        return param

    def _march(self):
        for param_name in self.param_names:
            d = self.direction[param_name[0]][param_name[1]]
            self.param[param_name[0]][param_name[1]] += d
        return self._get_current()

    def suggest(self) -> LearningParmeter:
        param_string = '.'.join(self.param_names[self.current_param_index-1])
        print(f'suggest state: {self.state} ({param_string})')
        if self.state == 'init':
            return self._get_current()
        elif self.state == 'evaluate':
            # Reset prev direction
            if self.current_param_index > 0:
                param_name = self.param_names[self.current_param_index-1]
                self.param[param_name[0]][param_name[1]] -= 1
            param_name = self.param_names[self.current_param_index]
            self.param[param_name[0]][param_name[1]] += 1
            return self._get_current()
        elif self.state == 'march':
            return self._march()
        else:
            raise Exception(f'Unknown state {self.state}')

    def feedback(self, loss: float):
        if self.state == 'init':
            self.last_loss = loss
            self.state = 'evaluate'
            self.current_param_index = 0
        elif self.state == 'evaluate':
            param_name = self.param_names[self.current_param_index]
            d = 1 if self.last_loss > loss else -1
            self.direction[param_name[0]][param_name[1]] = d
            self.last_loss = loss
            self.current_param_index += 1
            if self.current_param_index >= len(self.param_names):
                self.state = 'march'

        elif self.state == 'march':
            if self.last_loss > loss:
                self.last_loss = loss
            else:
                self.state = 'evaluate'
                self.current_param_index = -1
        else:
            raise Exception(f'Unknown state {self.state}')


class ParameterConstraint:
    def __init__(self, path, min, max, step_size):
        self.path = path
        self.min = min
        self.max = max
        self.step_size = step_size


class GridSearch:
    def __init__(self, constraints: List[ParameterConstraint], init: LearningParmeter = LearningParmeter('LstmTreeTagger')):
        self.param = {
            'common': init.__dict__,
            'model': init.model_hyper_parameter
        }
        self.constraints = constraints
        self.current_param_index = 0

        self.best_combination = self._get_current()
        self.best_loss = None

    def _get_current(self):
        param = LearningParmeter('LstmTreeTagger')
        param.__dict__ = self.param['common']
        param.model_hyper_parameter = self.param['model']
        return param

    def _increase(self, index) -> bool:
        '''
        Increase the param with the specified index if possible
        or reset it and increase the next on recursive.
        '''
        if index >= len(self.constraints):
            return False

        param_info = self.constraints[index]
        prev = self.param[param_info.path[0]][param_info.path[1]]
        if prev >= param_info.max:
            self.param[param_info.path[0]][param_info.path[1]] = param_info.min
            return self._increase(index+1)
        else:
            self.param[param_info.path[0]
                       ][param_info.path[1]] += param_info.step_size
            return True

    def suggest(self) -> LearningParmeter:

        if not self._increase(0):
            print(f'Finish')
            return None

        return self._get_current()

    def feedback(self, loss: float):
        if self.best_loss is None:
            self.best_loss = loss
            self.best_combination = deepcopy(self._get_current())
        elif self.best_loss > loss:
            self.best_loss = loss
            self.best_combination = deepcopy(self._get_current())

    @property
    def best(self):
        return self.best_combination
