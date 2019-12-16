import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


class FullyConnectedTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, blueprint, hyper_parameter):
        super(FullyConnectedTagger, self).__init__()
        # Config
        self.config = {
            'max_spread': 2,
            'embedding_size': 16,
            'spread': 2,
        }
        self.config.update(hyper_parameter)

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=self.config['embedding_size'],
            padding_idx=pad_token
        )

        self.blueprint = blueprint
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
        return F.relu(self.out(torch.cat((r, applied_attn), dim=1)))

    def forward(self, x, s):
        input = [p.view(-1) for p in torch.split(x, 1, dim=1)]
        input = [self.embedding(c) for c in input]

        # batch x sequence x embedding
        hidden = []
        for inst in self.blueprint:
            r, c = inst.get(input, hidden)
            hidden.append(self._cell(r, c))

        x = hidden[-1]
        x = self.hidden_to_tag(x)
        x = F.log_softmax(x, dim=1)
        return x

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
