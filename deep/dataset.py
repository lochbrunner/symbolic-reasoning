import numpy as np

from torch.utils.data import Dataset
import torch

from deep.generate import create_samples_permutation
from common.utils import memoize
from deep.node import Node


def scenarios_choices():
    return ['permutation']


class ScenarioParameter:
    def __init__(self, scenario: str, depth: int, spread: int):
        self.scenario = scenario
        self.depth = depth
        self.spread = spread


def ident_to_id(node: Node):
    # 0 is padding
    return ord(node.ident) - 97 + 1


class PermutationDataset(Dataset):

    def __init__(self, params, transform=None):
        self.transform = transform
        self.samples, self.idents, self.classes = create_samples_permutation(
            depth=params.depth, spread=params.spread)

        # Preprocess
        self.samples = [self._process_sample(sample) for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    @memoize
    def __getitem__(self, index):
        return self.samples[index]
        # return _process_sample(index)

    def _process_sample(self, sample):
        [y, x] = sample
        s = 2  # spread
        if self.transform is not None:
            return self.transform(x, y, s)
        return x, y, s

    @property
    def vocab_size(self):
        return len(self.idents)

    @property
    def tag_size(self):
        return len(self.classes)


class TraverseInstructionSet:
    def __init__(self, input=None, hidden=None):
        self.input = input
        self.hidden = hidden

    def get(self, input, hidden):
        if self.input is not None:
            return input[self.input]
        if self.hidden is not None:
            return hidden[self.hidden]
        raise Exception('Missing index!')

    def __repr__(self):
        if self.input is not None:
            return f'i{self.input}'
        if self.hidden is not None:
            return f'h{self.hidden}'
        raise Exception('Missing index!')


class TraverseInstruction:
    def __init__(self, root, childs):
        self.root = root
        self.childs = childs

    def get(self, input, hidden):
        root = self.root.get(input, hidden)
        childs = [child.get(input, hidden) for child in self.childs]
        return root, childs

    def __repr__(self):
        childs = ', '.join([repr(child) for child in self.childs])
        return f'{repr(self.root)} - {childs}'


class Embedder:
    '''Traversing post order

    Assuming each sample has the same form
    '''

    def __call__(self, x: Node, y, s):
        stack = [x]
        x = []
        seen = set()
        while len(stack) > 0:
            n = stack[-1]
            if len(n.childs) > 0 and n not in seen:
                stack.extend(n.childs[::-1])
                seen.add(n)
            else:
                stack.pop()
                x.append(ident_to_id(n))

        return x, y, s

    @staticmethod
    def blueprint(params):
        depth = params.depth
        spread = params.spread
        # One line for each node (not leaf)
        lines = [l for l in range(depth)[::-1] for i in range(spread**l)]
        i = 0
        h = 0
        instructions = []
        for l in lines:
            if l == depth-1:
                instructions.append(TraverseInstruction(
                    TraverseInstructionSet(input=i + spread),
                    [TraverseInstructionSet(input=i+s) for s in range(spread)]))
                i += spread + 1
            else:
                instructions.append(TraverseInstruction(
                    TraverseInstructionSet(input=i),
                    [TraverseInstructionSet(hidden=h+s) for s in range(spread)]))
                i += 1
                h += spread
        return instructions


class Padder:
    def __init__(self, depth=2, spread=2, pad_token=0):
        self.max_length = sum(
            [spread**l for l in range(0, depth+1)])
        self.pad_token = pad_token

    def __call__(self, x, y, s):
        padded_x = np.ones((self.max_length))*self.pad_token
        padded_x[0:len(x)] = x

        x = torch.as_tensor(padded_x, dtype=torch.long)  # pylint: disable=no-member
        y = torch.as_tensor(y, dtype=torch.long)  # pylint: disable=no-member

        return x, y, s


class Uploader:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, y, s):
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y, s
