import numpy as np

from torch.utils.data import Dataset
import torch

from deep.generate import create_samples_permutation
from common.utils import memoize
from deep.node import Node
# from deep.model import TrivialTreeTaggerUnrolled


def ident_to_id(node: Node):
    # 0 is padding
    return ord(node.ident) - 97 + 1


class PermutationDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        self.samples, self.idents, self.classes = create_samples_permutation(
            depth=2, spread=2)

    def __len__(self):
        return len(self.samples)

    @memoize
    def __getitem__(self, index):
        [y, x] = self.samples[index]
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


class Embedder:
    '''Traversing post order

    Assuming each sample has the same form
    '''

    def __call__(self, x: Node, y, s):
        x = [
            ident_to_id(x.childs[0]),
            ident_to_id(x.childs[1]),
            ident_to_id(x)
        ]
        return x, y, s


class Padder:
    def __init__(self, max_length=3, pad_token=0):
        self.max_length = max_length
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
