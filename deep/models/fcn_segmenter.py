import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


class FullyConnectedSegmenter(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, blueprint, hyper_parameter):
        super(FullyConnectedSegmenter, self).__init__()
        # Config
        self.config = {
            'max_spread': 2,
            'embedding_size': 16,
            'spread': 2,
        }
        self.config.update(hyper_parameter)
        self.blueprint = blueprint
        self.tagset_size = tagset_size

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=self.config['embedding_size'],
            padding_idx=pad_token
        )

        # Linear supports batch processing
        self.attn = nn.Linear(self.config['embedding_size'], self.config['embedding_size'])
        self.out = nn.Linear((1 + self.config['spread'])*self.config['embedding_size'],
                             self.config['embedding_size'])

        # To tag
        self.hidden_to_tag = nn.Linear(self.config['embedding_size'], tagset_size)

    def _cell(self, r, cs):
        # r: batch x embedding
        # cs: batch x spread x embedding
        cs = torch.stack(cs, dim=1)
        cs = F.relu(cs)
        r = F.relu(r)
        attn = self.attn(r)
        applied_attn = attn[:, None, :]*cs
        applied_attn = F.relu(applied_attn).view(-1, self.config['embedding_size']*2)
        h = F.relu(self.out(torch.cat((r, applied_attn), dim=1)))
        y = self.hidden_to_tag(h)
        y = F.log_softmax(y, dim=1)
        return h, y

    def forward(self, x, s):
        device = next(self.parameters()).device
        batch_size = x.size(0)
        seq_size = x.size(1)
        input = [p.view(-1) for p in torch.split(x, 1, dim=1)]
        input = [self.embedding(c) for c in input]

        y = torch.zeros(batch_size, self.tagset_size, seq_size).to(device)

        hidden = []
        for inst in self.blueprint:
            r, c = inst.get(input, hidden)
            h, yy = self._cell(r, c)
            hidden.append(h)
            i = inst.get_index()
            y[:, :, i] = yy
        return y

    def _cell_introspect(self, x, embedder):
        r = torch.as_tensor(embedder(x))
        r = self.embedding(r)

        if len(x.childs) == 0:
            return r
        cs = [self._cell_introspect(child, embedder) for child in x.childs]

        cs = torch.stack(cs, dim=0)
        cs = F.relu(cs)
        r = F.relu(r)
        attn = self.attn(r)
        applied_attn = attn[None, :]*cs
        applied_attn = F.relu(applied_attn).view(self.config['embedding_size']*2)
        return F.relu(self.out(torch.cat((r, applied_attn), dim=0)))

    def introspect(self, x, embedder):
        introspection = {}
        x = self._cell_introspect(x, embedder)

        x = self.hidden_to_tag(x)
        x = F.log_softmax(x, dim=0)
        introspection['scores'] = x
        return introspection

    def activation_names(self):
        return ['scores']
