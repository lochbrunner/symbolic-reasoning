class LearningParmeter:
    def __init__(self, num_epochs: int = 10, learning_rate: float = 0.1,
                 batch_size: int = 10, gradient_clipping: float = 0.1,
                 model_hyper_parameter: dict = {}):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.model_hyper_parameter = model_hyper_parameter


class Independent:
    '''
    Assumes that the parameter are roughly independent.

    Evaluation policy (finding new direction)
    * begin
    * increasing step
    * a) fixed period
    * b) when things went worse
    '''

    def __init__(self, init: LearningParmeter = LearningParmeter()):
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
        param = LearningParmeter()
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
