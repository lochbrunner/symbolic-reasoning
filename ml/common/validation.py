import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.terminal_utils import printHistogram


class Ratio:
    def __init__(self, size=10, count_print=3):
        assert size >= count_print
        self.tops = np.zeros(size, dtype=np.uint16)
        self.sum = 0
        self.size = size
        self.count_print = count_print

    def topk(self, k):
        nom = np.sum(self.tops[:k])
        return float(nom) / max(1, float(self.sum))

    def __float__(self):
        return self.topk(1)

    def __str__(self):
        v = [(1. - self.topk(i+1)) * 100 for i in range(self.count_print)]
        v = [f'{i:.1f}%' for i in v]
        remaining = ', '.join(v[1:])
        return f'{v[0]}% ({remaining})%)'

    def update(self, mask, predict, truth):
        '''
        n: node
        r: rule
        mask: n
        truth: n
        predict: r, n
        '''

        predict = (-predict).argsort(axis=0)
        if mask is not None:
            predict = predict[:, mask]
            truth = truth[mask]
            self.sum += np.sum(mask).item()
        else:
            self.sum += truth.shape[0]

        for i in range(self.size):
            p = predict[i]
            self.tops[i] += np.sum(p == truth)

    def update_global(self, mask, predict, truth):
        '''
        n: node
        r: rule
        mask: n
        truth: n
        predict: r, n
        '''
        if mask is not None:
            predict = predict[:, mask]
            truth = truth[mask]
        # Find
        n = predict.shape[1]
        predict = (-predict).flatten().argsort()

        truth_path = np.argmax(truth)
        truth_rule_id = truth[truth_path]
        encoded_id = truth_rule_id * n + truth_path
        top = np.where(predict == encoded_id)[0][0]

        if top < self.size:
            self.tops[top] += 1
        self.sum += 1

    def printHistogram(self):
        printHistogram(self.tops, range(self.size), self.sum)

    def as_dict(self):
        return {'tops': self.tops.tolist(), 'total': self.sum}


class Error:
    def __init__(self, with_padding=None, when_rule=None, exact=None, exact_no_padding=None):
        self.with_padding = with_padding or Ratio()
        self.when_rule = when_rule or Ratio()
        self.exact = exact or Ratio()
        self.exact_no_padding = exact_no_padding or Ratio()


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: DataLoader):
    error = Error(exact=Ratio(20), exact_no_padding=Ratio(20))

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
            predicted_padding = np.copy(predict)
            predicted_padding[0, :] = np.finfo('f').min

            error.with_padding.update(None, predict, truth)
            error.when_rule.update((truth > 0), predict, truth)
            error.exact.update_global(None, predict, truth)
            error.exact_no_padding.update_global(None, predicted_padding, truth)

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
        self.assertEqual(ratio.topk(2), 1)

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
