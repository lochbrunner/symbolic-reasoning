#!/usr/bin/env python3

import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
import sys


class Mean:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.max = 0

    def __add__(self, correct):
        self.total += 1
        if isinstance(correct, bool):
            if correct:
                self.correct += 1.
        else:
            self.correct += correct
            self.max = max(self.max, correct)
        return self

    @property
    def summary(self):
        if self.total == 0:
            return -1.
        return self.correct / self.total

    @property
    def statistic(self):
        return f'Ø {self.summary:.1f} (max: {self.max})'

    @property
    def verbose(self):
        return f'{self} ({int(self.correct)} / {self.total})'

    def __str__(self):
        v = self.summary * 100
        return f'{v:.2f}%'

    def __float__(self):
        return self.summary


class Ratio:
    def __init__(self, size=10, count_print=10):
        assert size >= count_print
        self.tops = np.zeros(size, dtype=np.uint16)
        self.sum = 0
        self.size = size
        self.count_print = count_print

    def topk(self, k: int):
        '''k starts with 1 as it stands for the (first) best match ratio. '''
        nom = np.sum(self.tops[:k])
        return float(nom) / max(1, float(self.sum))

    def __float__(self):
        return self.topk(1)

    def __str__(self):
        v = [(1. - self.topk(i+1)) * 100 for i in range(self.count_print)]
        v = [f'{i:.1f}%' for i in v]
        remaining = ', '.join(v[1:])
        return f'{v[0]} ({remaining})'

    def log(self, logger, prefix):
        vs = [(1. - self.topk(i+1)) * 100 for i in range(self.count_print)]
        for i, v in enumerate(vs, 1):
            logger(f'{prefix} [{i}]', v)

    def log_bundled(self, logger, label, epoch):
        if logger is not None:
            vs = [(1. - self.topk(i+1)) for i in range(self.count_print)]
            scalars = {f'top [{i}]': v for i, v in enumerate(vs)}
            logger.add_scalars(label, scalars, epoch)

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

        for i in range(min(self.size, predict.shape[0])):
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

        # Multiple solutions (>0) could be valid
        truth_paths = truth.nonzero()[0]
        # Probably the negatives are masked out?
        if len(truth_paths) > 0:
            top = self.size
            for truth_path in truth_paths:
                truth_rule_id = truth[truth_path]
                encoded_id = truth_rule_id * n + truth_path
                top = min(top, np.where(predict == encoded_id)[0][0])

            if top < self.size:
                self.tops[top] += 1
            self.sum += 1

    # def printHistogram(self):
    #     printHistogram(self.tops, range(self.size), self.sum)

    def as_dict(self):
        return {'tops': self.tops.tolist(), 'total': self.sum}


class Error:
    def __init__(self, with_padding=None, when_rule=None, exact=None, exact_no_padding=None):
        self.with_padding = with_padding or Ratio()
        self.when_rule = when_rule or Ratio()
        self.exact = exact or Ratio()
        self.exact_no_padding = exact_no_padding or Ratio()
        self.value_all = Mean()
        self.value_positive = Mean()
        self.value_negative = Mean()

    def as_dict(self):
        return {'exact': self.exact.as_dict(),
                'exact-no-padding': self.exact_no_padding.as_dict(),
                'when-rule': self.when_rule.as_dict(),
                'with-padding': self.with_padding.as_dict(),
                'value-all': float(self.value_all),
                'value-positive': float(self.value_positive),
                'value-negative': float(self.value_negative),
                }

    def __str__(self):
        summary = ''
        summary += f'with padding: {self.with_padding}\n'
        summary += f'when rule: {self.when_rule}\n'
        summary += f'exact: {self.exact}\n'
        summary += f'exact no padding: {self.exact_no_padding}\n'
        summary += f'value all: {self.value_all}\n'
        summary += f'value positive: {self.value_positive}\n'
        summary += f'value negative: {self.value_negative}\n'
        return summary


@dataclass
class ValidationResult:
    error: Error
    predicted_rule_distribution: np.array
    policy_loss: float
    value_loss: float

    def __str__(self):
        return str(self.error)


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: DataLoader,
             policy_loss_function=None, value_loss_function=None, no_negative=True, show_progress=False) -> ValidationResult:
    '''Create validation result of all samples in the dataloader with the specified model.'''

    error = Error(exact=Ratio(20), exact_no_padding=Ratio(20))

    policy_loss = 0
    value_loss = 0
    predicted_rule_distribution = None

    for x, s, y, p, v in tqdm(dataloader, desc='validate', disable=not show_progress or not sys.stdin.isatty()):
        x = x.to(model.device)
        s = s.to(model.device)
        y = y.to(model.device)
        p = p.to(model.device)
        v = v.to(model.device)
        v = v.squeeze(dim=1)
        # Dimensions
        # x: batch * label * length
        # y: batch * length
        py, pv = model(x, s, p)
        # py: batch x rule x path
        if value_loss_function is not None:
            value_loss += value_loss_function(pv, v).item()
        if policy_loss_function is not None:
            policy_loss += policy_loss_function(py, y).item()
        if predicted_rule_distribution is None:
            predicted_rule_distribution = np.zeros(py.shape[1], dtype=int)
        assert py.size(0) == y.size(0), f'{py.size(0)} == {y.size(0)}'
        batch_size = py.size(0)
        py = np.exp(py.cpu().numpy())
        y = y.cpu().numpy()
        p = p.cpu().numpy()
        pv = pv.cpu().numpy()
        gt_v = v.cpu().numpy()

        ry = py.max(axis=2).argmax(axis=1)
        bins = np.bincount(ry, minlength=py.shape[1])
        predicted_rule_distribution += bins

        if no_negative:
            y = y*(p+1)/2

        for i in range(batch_size):
            # policy
            truth = y[i, :]
            predict = py[i, :, :]
            predicted_padding = np.copy(predict)
            predicted_padding[0, :] = np.finfo('f').min
            error.with_padding.update(None, predict, truth)
            error.when_rule.update((truth > 0), predict, truth)
            error.exact.update_global(None, predict, truth)
            error.exact_no_padding.update_global(None, predicted_padding, truth)

            # value
            # as the value head is using the log softmax we have to "un-log" it
            # 1-gt_v as we are interested in the error
            value_err = np.exp(pv[i, 1-gt_v[i]]).item()
            error.value_all += value_err
            if gt_v[i] == 0:
                error.value_negative += value_err
            else:
                error.value_positive += value_err

    return ValidationResult(error=error, policy_loss=policy_loss, value_loss=value_loss,
                            predicted_rule_distribution=predicted_rule_distribution)


class TestRatio(unittest.TestCase):
    '''Unit tests for python Ratio class'''

    def test_update(self):
        ratio = Ratio(3, 3)
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

    def test_update_global_all_multiples(self):
        mask = np.array([True, True])
        # three rules. two nodes
        # Rule 1 should be applied at 0
        # #rules * n
        predict = np.array([[0.95, 0.8], [1, 0.9], [0.85, 1.0]])

        # rule 2 @ 0 was predicted with 0.85 -> 5. best.
        truth = np.array([2, 0])
        ratio = Ratio()
        ratio.update_global(mask, predict, truth)
        self.assertEqual(ratio.tops.tolist(), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

        # rule 1 @ 1 was predicted with 0.9 -> 4. best.
        truth = np.array([0, 1])
        ratio = Ratio()
        ratio.update_global(mask, predict, truth)
        self.assertEqual(ratio.tops.tolist(), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

        # The best of both -> 4
        truth = np.array([2, 1])
        ratio = Ratio()
        ratio.update_global(mask, predict, truth)
        self.assertEqual(ratio.tops.tolist(), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_update_global_single_symbol(self):
        ratio = Ratio()
        # rules * length
        mask = np.array([True])
        correct_predict = np.array([[0.9], [1]])
        truth = np.array([1])
        ratio.update_global(mask, correct_predict, truth)
        self.assertEqual(ratio.topk(1), 1.0)
        self.assertEqual(ratio.topk(2), 1.0)

        ratio = Ratio()
        wrong_predict = np.array([[1.], [0.9]])
        ratio.update_global(mask, wrong_predict, truth)
        self.assertEqual(ratio.topk(1), 0.0)
        self.assertEqual(ratio.topk(2), 1.0)


class Validation(unittest.TestCase):
    class ModelStub:
        def __init__(self, py=None, pv=None):
            self.py = py or [[[0., 0., 0.],  # Padding
                              [0., 1., 0.],  # rule 1
                              [.5, 0., 0.]   # rule 2
                              ]]
            self.pv = pv or [[1., 1.e-5]]

        def __call__(self, x, s, p):
            # batch, rule, path
            py = np.array(self.py)
            pv = np.array(self.pv)
            return torch.from_numpy(py), torch.from_numpy(np.log(pv))

        @property
        def device(self):
            return 'cpu'

    def test_perfect_positive_with_negative(self):
        # x, s, y, p, v
        x = torch.as_tensor([])  # Just eaten by the model stub
        s = torch.as_tensor([])  # Just eaten by the model stub
        y = torch.as_tensor([[0, 1, 0]])
        p = torch.as_tensor([[0., +1., 0.]])
        v = torch.as_tensor([[1]])
        dataloader = [(x, s, y, p, v)]

        validation = validate(model=Validation.ModelStub(), dataloader=dataloader, no_negative=False)

        self.assertEqual(validation.error.with_padding.tops.tolist(), [2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.when_rule.tops.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.exact.tops.tolist(), [1]+[0]*19)
        self.assertEqual(validation.error.exact_no_padding.tops.tolist(), [1]+[0]*19)

        self.assertEqual(float(validation.error.value_all), 1.)
        self.assertEqual(float(validation.error.value_positive), 1.)
        self.assertEqual(float(validation.error.value_negative), -1.)  # Not defined

    def test_perfect_positive_without_negative(self):
        # x, s, y, p, v
        x = torch.as_tensor([])  # Just eaten by the model stub
        s = torch.as_tensor([])  # Just eaten by the model stub
        y = torch.as_tensor([[0, 1, 0]])
        p = torch.as_tensor([[0., +1., 0.]])
        v = torch.as_tensor([[1]])
        dataloader = [(x, s, y, p, v)]

        validation = validate(model=Validation.ModelStub(), dataloader=dataloader, no_negative=True)
        self.assertEqual(validation.error.with_padding.tops.tolist(), [2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.when_rule.tops.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.exact.tops.tolist(), [1]+[0]*19)
        self.assertEqual(validation.error.exact_no_padding.tops.tolist(), [1]+[0]*19)

        self.assertEqual(float(validation.error.value_all), 1.)
        self.assertEqual(float(validation.error.value_positive), 1.)
        self.assertEqual(float(validation.error.value_negative), -1.)  # Not defined

    def test_perfect_negative_with_negative(self):
        # x, s, y, p, v
        x = torch.as_tensor([])  # Just eaten by the model stub
        s = torch.as_tensor([])  # Just eaten by the model stub
        y = torch.as_tensor([[2, 0, 0]])
        p = torch.as_tensor([[0., -1., 0.]])
        v = torch.as_tensor([[1]])
        dataloader = [(x, s, y, p, v)]
        py = [[[0., 0., 0.],  # Padding
               [0., -1., 0.],  # rule 1
               [.5, 0., 0.]   # rule 2
               ]]

        validation = validate(model=Validation.ModelStub(py=py), dataloader=dataloader, no_negative=False)

        self.assertEqual(validation.error.with_padding.tops.tolist(), [3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.when_rule.tops.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.exact.tops.tolist(), [1]+[0]*19)
        self.assertEqual(validation.error.exact_no_padding.tops.tolist(), [1]+[0]*19)

        self.assertEqual(float(validation.error.value_all), 1.)
        self.assertEqual(float(validation.error.value_positive), 1.)
        self.assertEqual(float(validation.error.value_negative), -1.)  # Not defined

    def test_positives_not_perfect_with_negatives(self):
        # x, s, y, p, v
        x = torch.as_tensor([])  # Just eaten by the model stub
        s = torch.as_tensor([])  # Just eaten by the model stub
        y = torch.as_tensor([[2, 0, 0]])
        p = torch.as_tensor([[1., 0., 0.]])
        v = torch.as_tensor([[1]])
        dataloader = [(x, s, y, p, v)]
        py = [[[0., 0., 0.8],  # Padding
               [0., 1., 0.8],  # rule 1
               [.5, 0., 0.8]   # rule 2
               ]]

        validation = validate(model=Validation.ModelStub(py=py), dataloader=dataloader, no_negative=False)

        self.assertEqual(validation.error.with_padding.tops.tolist(), [2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.when_rule.tops.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.exact.tops.tolist(), [0, 0, 0, 0, 1]+[0]*15)
        self.assertEqual(validation.error.exact_no_padding.tops.tolist(), [0, 0, 0, 1, 0]+[0]*15)

    def test_positives_not_perfect_without_negatives(self):
        # x, s, y, p, v
        x = torch.as_tensor([])  # Just eaten by the model stub
        s = torch.as_tensor([])  # Just eaten by the model stub
        y = torch.as_tensor([[2, 0, 0]])
        p = torch.as_tensor([[1., 0., 0.]])
        v = torch.as_tensor([[1]])
        dataloader = [(x, s, y, p, v)]
        py = [[[0., 0., 0.8],  # Padding
               [0., 1., 0.8],  # rule 1
               [.5, 0., 0.8]   # rule 2
               ]]

        validation = validate(model=Validation.ModelStub(py=py), dataloader=dataloader, no_negative=True)

        self.assertEqual(validation.error.with_padding.tops.tolist(), [2, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.when_rule.tops.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(validation.error.exact.tops.tolist(), [0, 0, 0, 0, 1]+[0]*15)
        self.assertEqual(validation.error.exact_no_padding.tops.tolist(), [0, 0, 0, 1, 0]+[0]*15)

        # print(f'with_padding: {validation.error.with_padding.tops.tolist()}')
        # print(f'when_rule: {validation.error.when_rule.tops.tolist()}')
        # print(f'exact: {validation.error.exact.tops.tolist()}')
        # print(f'exact_no_padding: {validation.error.exact_no_padding.tops.tolist()}')


if __name__ == '__main__':
    unittest.main()
