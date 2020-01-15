import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn


class RnnTreeTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, pad_token, blueprint, spread, hyper_parameter, Rnn):
        super(RnnTreeTagger, self).__init__()
        self.blueprint = blueprint
        self.config = {
            'embedding_size': 32,
            'lstm_hidden_size': 64,
            'lstm_layers': 1
        }
        self.config.update(hyper_parameter)
        self.spread = spread

        # Embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size+1,
            embedding_dim=self.config['embedding_size'],
            padding_idx=pad_token
        )

        # Rnn
        self.lstm = Rnn(
            input_size=self.config['embedding_size'],
            hidden_size=self.config['lstm_hidden_size'],
            num_layers=self.config['lstm_layers'],
            batch_first=True,
        )

        lstm_h = torch.empty(self.config['lstm_layers'],  # pylint: disable=no-member
                             self.config['lstm_hidden_size'])
        nn.init.uniform_(lstm_h, -1., 1.)
        self.lstm_h = nn.Parameter(lstm_h)

        lstm_c = torch.empty(self.config['lstm_layers'],  # pylint: disable=no-member
                             self.config['lstm_hidden_size'])
        nn.init.uniform_(lstm_c, -1., 1.)
        self.lstm_c = nn.Parameter(lstm_c)

        # Combine ident with childs
        self.combine = nn.Linear(
            self.config['lstm_hidden_size']+self.config['embedding_size'], self.config['embedding_size'])

        # To tag
        self.hidden_to_tag = nn.Linear(self.config['embedding_size'], tagset_size)

    def _cell(self, r, cs, introspection):
        c = torch.stack(cs, dim=1)  # pylint: disable=no-member
        batch_size = r.size(0)
        # batch x sequence x embedding
        c = rnn.pack_padded_sequence(c, self.spread, batch_first=True, enforce_sorted=False)
        lstm_h = self.lstm_h[:, None].expand(-1, batch_size, -1)
        lstm_c = self.lstm_c[:, None].expand(-1, batch_size, -1)
        c, _ = self.lstm(c, (lstm_h, lstm_c))
        c, _ = rnn.pad_packed_sequence(c, batch_first=True)

        # Combine root with label
        c = F.relu(c)
        c = torch.split(c, 1, dim=1)[-1].view(-1, self.config['lstm_hidden_size'])
        x = torch.cat((r, c), dim=1)
        x = self.combine(x)
        x = F.relu(x)

        return x

    def forward(self, x, *args, introspection=None):
        # Expect for spread=2 and depth=2:
        #  i2, (i0, i1)
        #  i5, (i3, i4)
        #  i6, (h0, h1)

        # x: batch x sequence
        input = [p.view(-1) for p in torch.split(x, 1, dim=1)]
        input = [self.embedding(c) for c in input]

        hidden = []
        for inst in self.blueprint:
            r, c = inst.get(input, hidden)
            hidden.append(self._cell(r, c, introspection))

        x = hidden[-1]
        x = self.hidden_to_tag(x)
        x = F.log_softmax(x, dim=1)
        if introspection is not None:
            introspection['scores'] = x.detach()
        return x

    def _cell_introspect(self, x, embedder):
        r = torch.as_tensor(embedder(x))
        r = self.embedding(r)
        if len(x.childs) == 0:
            return r
        childs = [self._cell_introspect(child, embedder) for child in x.childs]
        childs = torch.stack(childs, dim=0)[None, :]

        lstm_h = self.lstm_h[None, :]
        lstm_c = self.lstm_c[None, :]
        c, _ = self.lstm(childs, (lstm_h, lstm_c))

        # Combine root with label
        c = F.relu(c)
        c = torch.split(c, 1, dim=1)[-1].view(self.config['lstm_hidden_size'])
        x = torch.cat((r, c), dim=0)
        x = self.combine(x)
        x = F.relu(x)

        return x

    # Should this be moved into sub class?

    def introspect(self, x, embedder):
        '''
        Infers the node and returns 'all' intermediate results
        '''
        introspection = {}
        x = self._cell_introspect(x, embedder)

        x = self.hidden_to_tag(x)
        x = F.log_softmax(x, dim=0)
        introspection['scores'] = x

        return introspection

    def activation_names(self):
        return ['scores']


class LstmTreeTagger(RnnTreeTagger):
    def __init__(self, vocab_size, tagset_size, pad_token, blueprint, hyper_parameter):
        super(LstmTreeTagger, self).__init__(vocab_size, tagset_size,
                                             pad_token, blueprint, hyper_parameter, nn.LSTM)


class GruTreeTagger(RnnTreeTagger):
    def __init__(self, vocab_size, tagset_size, pad_token, blueprint, hyper_parameter):
        super(GruTreeTagger, self).__init__(vocab_size, tagset_size,
                                            pad_token, blueprint, hyper_parameter, nn.GRU)
