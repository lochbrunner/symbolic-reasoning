#!/usr/bin/env python3

from itertools import permutations
from string import ascii_lowercase as alphabet
from random import choices, shuffle

import torch
import torch.nn as nn
import torch.optim as optim

from pycore import SymbolBuilder
from ml_common import predict_rule, create_batches
from models import IdentTreeLstmModeler, RootIdentModeler, RnnModeler


def create_samples(level=1, spread=2, noise=2):
    '''Samples are of the form (class_id, symbol)'''
    samples = []

    size = spread**level

    builder = SymbolBuilder()
    for _ in range(level):
        builder.add_level_uniform(spread)

    idents = alphabet[:size]
    classes = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)
        builder.set_level_idents(level, perm)

        # Add noise to the upper nodes
        for _ in range(noise):
            for l in range(level):
                level_size = spread**l
                level_idents = choices(idents, k=level_size)
                builder.set_level_idents(l, level_idents)

            samples.append((i, builder.symbol))

    return samples, idents, classes


@torch.no_grad()
def validate_sample(predict_rule, samples, rules_size):
    '''Validates the model based on the samples.

    Returns a list [fraction of top 1, top2, ...]
    '''
    ranks = [0] * rules_size

    for id, sample in samples:
        log_probs, _ = predict_rule(sample)
        probs = log_probs.exp().flatten().tolist()

        s = sorted([(v, i) for i, v in enumerate(probs)], reverse=True)
        ixs = [i for v, i in s]
        pos = ixs.index(id)
        ranks[pos] += 1

    samples_size = len(samples)
    return [rank / samples_size for rank in ranks]


def main():
    torch.manual_seed(1)
    samples, idents, classes = create_samples(level=1, noise=100)
    ident_to_ix = {ident: i for i, ident in enumerate(idents)}
    class_to_ix = {i: i for i, _ in enumerate(classes)}

    print(f'#classes: {len(classes)}    #idents: {len(idents)}')

    shuffle(samples)

    BATCH_SIZE = 8
    batches = create_batches(samples, BATCH_SIZE)

    # model = IdentTreeLstmModeler(ident_size=len(idents),
    #                              remainder_dim=8,
    #                              hidden_dim=8,
    #                              embedding_dim=8,
    #                              rules_size=len(classes))

    # model = RootIdentModeler(len(idents), 32, len(classes))

    model = RnnModeler(len(idents), 16, 16, 16, [16, 16], len(classes))

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    # Recording
    ranks = []

    for epoch in range(21):
        total_loss = 0
        for i, batch in enumerate(batches):
            model.zero_grad()
            for (id, sample) in batch:
                log_probs, _ = predict_rule(model, sample, ident_to_ix)
                log_probs = log_probs.view(1, -1)
                expected = torch.tensor([id], dtype=torch.long)
                loss = loss_function(log_probs, expected)
                loss.backward()
                total_loss += loss.item()
            optimizer.step()

        # Validate
        rank = validate_sample(lambda part: predict_rule(model, part, ident_to_ix), samples, len(classes))
        ranks.append(rank)
        error_pc = (1.-rank[0])*100.0
        print(f'[{epoch}] Error: {error_pc:.3}%  loss: {total_loss:.6}')


if __name__ == '__main__':
    main()
