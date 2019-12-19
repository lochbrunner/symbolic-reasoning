#!/usr/bin/env python3
from .parameter_search import *
import unittest


@unittest.skip("no asserts only manual test")
class TestMarchSearch(unittest.TestCase):
    def test_2floats(self):
        def model(param: LearningParmeter):
            x = param.learning_rate
            y = param.gradient_clipping
            return ((10.-x)**2. + (10.-y)**2.)

        init = LearningParmeter(10, 0., 10, 0.0, {})

        solver = MarchSearch(init)

        for _ in range(30):
            param = solver.suggest()
            print(
                f'Param x: {param.learning_rate} y: {param.gradient_clipping} loss: {model(param)}')
            loss = model(param)

            solver.feedback(loss)


class TestGridSearch(unittest.TestCase):
    def test_2floats(self):
        def model(param: LearningParmeter):
            x = param.learning_rate
            y = param.gradient_clipping
            return ((0.3-x)**2. + (0.1-y)**2.)

        init = LearningParmeter(10, 0., 10, 0.0, {})
        constaints = [
            ParameterConstraint(['common', 'num_epochs'], 10, 10, 1),
            ParameterConstraint(['common', 'learning_rate'], 0.1, 0.5, 0.1),
            ParameterConstraint(['common', 'batch_size'], 10, 20, 15),
            ParameterConstraint(
                ['common', 'gradient_clipping'], 0.05, 0.20, 0.05),
            ParameterConstraint(
                ['model', 'embedding_size'], 8, 16, 4),
        ]

        solver = GridSearch(constaints, init)

        for _ in range(30):
            param = solver.suggest()
            if param is None:
                break
            loss = model(param)
            # print(
            #     f'loss: {loss} @ ({param.learning_rate}, {param.gradient_clipping})')

            solver.feedback(loss)

        self.assertAlmostEqual(solver.best.learning_rate, 0.3)
        self.assertAlmostEqual(solver.best.gradient_clipping, 0.1)


if __name__ == '__main__':
    unittest.main()
