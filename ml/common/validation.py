import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader


class Ratio:
    def __init__(self):
        self.first = 0
        self.second = 0
        self.third = 0
        self.sum = 0

    def __float__(self):
        return float(self.first) / max(1, float(self.sum))

    def top_2(self):
        '''Relevant for beam size'''
        return float(self.first + self.second) / max(1, float(self.sum))

    def top_3(self):
        '''Relevant for beam size'''
        return float(self.first + self.second + self.third) / max(1, float(self.sum))

    def __str__(self):
        v = (1. - float(self)) * 100.
        vs = (1. - self.top_2()) * 100.
        vt = (1. - self.top_3()) * 100.
        return f'{v:.1f}% ({vs:.1f}%, {vt:.1f}%)'

    def update(self, mask, predict, truth):
        '''
        n: node
        r: rule
        mask: n
        truth: n
        predict: r, n
        '''

        predict = (-predict).argsort(axis=0)

        pf = predict[0]
        pf = pf[mask]
        ps = predict[1]
        ps = ps[mask]
        pt = predict[2]
        pt = pt[mask]
        t = truth[mask]

        self.first += np.sum(pf == t)
        self.second += np.sum(ps == t)
        self.third += np.sum(pt == t)
        self.sum += np.sum(mask)

    def update_global(self, mask, predict, truth):
        '''
        n: node
        r: rule
        mask: n
        truth: n
        predict: r, n
        '''
        # Find
        predict = predict[:, mask]
        n = predict.shape[1]
        predict = (-predict).flatten().argsort()

        truth = truth[mask]
        truth_path = np.argmax(truth)
        truth_rule_id = truth[truth_path]
        encoded_id = truth_rule_id * n + truth_path
        top = np.where(predict == encoded_id)[0][0]
        print(f'top: {top}')

        if top == 0:
            self.first += 1
        elif top == 1:
            self.second += 1
        elif top == 2:
            self.third += 1
        self.sum += 1


class Error:
    def __init__(self, with_padding=None, when_rule=None, exact=None):
        self.with_padding = with_padding or Ratio()
        self.when_rule = when_rule or Ratio()
        self.exact = exact or Ratio()


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: DataLoader):
    error = Error()

    for x, y in dataloader:
        x = x.to(model.device)
        y = y.to(model.device)
        # Dimensions
        # x: batch * label * length
        # y: batch * length
        x = model(x)
        assert x.size(0) == y.size(0), f'{x.size(0)} == {y.size(0)}'
        batch_size = x.size(0)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        for i in range(batch_size):
            truth = y[i, :]
            predict = x[i, :, :]

            error.with_padding.update((truth > -1), predict, truth)
            error.when_rule.update((truth > 0), predict, truth)
            error.exact.update_global((truth > -1), predict, truth)

    return error


class TestRatio(unittest.TestCase):
    '''Unit tests for python Ratio class'''

    def test_update(self):
        ratio = Ratio()
        # n = 2, r = 3
        predict = np.array([[0, 1], [1, 0], [0, 0]])
        truth = np.array([1, 0])
        mask = np.array([True, True])
        ratio.update(mask, predict, truth)
        self.assertEqual(float(ratio), 1)

        truth = np.array([1, 1])
        ratio.update(mask, predict, truth)
        self.assertEqual(float(ratio), 0.75)
        self.assertEqual(ratio.top_2(), 1)

        truth = np.array([0, 1])
        ratio.update(mask, predict, truth)
        self.assertEqual(float(ratio), 0.5)

    def test_update_global(self):
        ratio = Ratio()
        # three rules. two nodes
        # Rule 1 should be applied at 0
        predict = np.array([[0, 0], [1, 0], [0, 0]])
        truth = np.array([1, 0])
        mask = np.array([True, True])
        ratio.update_global(mask, predict, truth)
        self.assertEqual(float(ratio), 1)


if __name__ == '__main__':
    unittest.main()
