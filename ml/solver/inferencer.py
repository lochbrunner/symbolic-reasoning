
import numpy as np
import torch

# project
from dataset.transformers import Padder, Embedder, ident_to_id
from common import io


class Inferencer:
    ''' Standard inferencer for unique index map per sample
    '''

    def __init__(self, model_filename: str):
        self.model, snapshot = io.load_model(model_filename)
        self.model.eval()
        # Copy of BagDataset
        self.ident_dict = {ident: (value+1) for (value, ident) in enumerate(snapshot['idents'])}
        self.spread = snapshot['kernel_size'] - 2
        self.pad_token = snapshot['pad_token']

    def __call__(self, initial, count=None):
        # x, s, _ = self.dataset.embed_custom(initial)
        x, s, _, _, _ = initial.embed(self.ident_dict, self.pad_token, self.spread, [], True)
        x = torch.unsqueeze(torch.as_tensor(np.copy(x), device=self.model.device), 0)
        s = torch.unsqueeze(torch.as_tensor(np.copy(s), device=self.model.device), 0)
        p = torch.ones(x.shape[:-1])

        y, v = self.model(x, s, p)
        y = y.squeeze()  # shape: rules, localisation
        y = y.cpu().detach().numpy()[1:, :-1]  # Remove padding
        value = v.cpu().detach().numpy()[0][0]
        value = np.exp(value)

        parts_path = [p[0] for p in initial.parts_bfs_with_path]
        i = (-y).flatten().argsort()

        def calc(n):
            p = np.unravel_index(i[n], y.shape)
            return p[0]+1, parts_path[p[1]], y[p[0], p[1]]  # rule at path

        if count is None:
            count = i.shape[0]

        return [calc(i) for i in range(count)], value


class SharedInferencer:
    '''Deprecated'''

    def __init__(self, model_filename, depth=None):
        self.model, snapshot = io.load_model(model_filename, depth=depth)
        self.idents = snapshot['idents']
        self.model.eval()
        self.spread = self.model.spread
        self.depth = self.model.depth

        self.paths = list(Embedder.legend(spread=self.spread, depth=self.depth)) + [None]

    def create_mask(self, node):
        def hash_path(path):
            if path is None:
                return '<padding>'
            return '/'.join([str(p) for p in path])
        path_ids = set()
        for path, _ in node.parts_dfs_with_path:
            path_ids.add(hash_path(path))

        return np.array([(hash_path(p) in path_ids) for p in self.paths])

    def __call__(self, node, count):
        assert self.depth >= node.depth, f'{self.depth} >= {node.depth}'
        x = Padder.pad(node, spread=self.spread, depth=self.depth)
        x = [ident_to_id(n, self.idents) for n in Embedder.unroll(x)] + [0]
        x = torch.as_tensor(x, dtype=torch.long, device=self.model.device)
        y = self.model(x.unsqueeze(0))

        y = y.squeeze()
        mask = self.create_mask(node)
        y = y.cpu().detach().numpy()[1:]  # Remove padding
        y = y[:, mask]
        paths = [path for i, path in enumerate(self.paths) if mask[i]]
        i = (-y).flatten().argsort()

        def calc(n):
            p = np.unravel_index(i[n], y.shape)
            return p[0]+1, paths[p[1]]  # rule at path

        return [calc(i) for i in range(count)]
