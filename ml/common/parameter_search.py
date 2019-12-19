from typing import List
from copy import deepcopy


class LearningParmeter:
    def __init__(self, model_name, num_epochs: int = 10, learning_rate: float = 0.1,
                 batch_size: int = 10, gradient_clipping: float = 0.1,
                 model_hyper_parameter: dict = {}):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.model_hyper_parameter = model_hyper_parameter


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
