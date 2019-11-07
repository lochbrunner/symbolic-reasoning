#!/usr/bin/env python3
from common.parameter_search import Independent, LearningParmeter
import unittest


class TestParameterSearch(unittest.TestCase):
    def test_2floats(self):
        def model(param: LearningParmeter):
            x = param.learning_rate
            y = param.gradient_clipping
            return ((10.-x)**2. + (10.-y)**2.)

        init = LearningParmeter(10, 0., 10, 0.0, {})

        solver = Independent(init)

        for _ in range(30):
            param = solver.suggest()
            print(
                f'Param x: {param.learning_rate} y: {param.gradient_clipping} loss: {model(param)}')
            loss = model(param)

            solver.feedback(loss)


if __name__ == '__main__':
    unittest.main()
