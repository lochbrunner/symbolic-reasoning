#!/usr/bin/env python3

import sys
from pycore import Trace, Symbol, Rule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_trace():
    file = '../out/trace.bin'

    try:
        trace = Trace.load(file)
    except Exception as e:
        print(f'Error loading {file}: {e}')
        sys.exit(1)
    return trace


class IdentModeler(nn.Module):
    """
    Using fully connected layer
    """

    def __init__(self, ident_size, embedding_dim, rules_size):
        super(IdentModeler, self).__init__()
        self.embeddings = nn.Embedding(ident_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, rules_size)

    def forward(self, prev_symbol):
        """
        Predicts the rule to apply
        """
        embeds = self.embeddings(prev_symbol).view((1, -1))
        h = F.relu(self.linear1(embeds))
        h = self.linear2(h)
        return F.log_softmax(h, dim=1)


if __name__ == "__main__":
    trace = load_trace()

    torch.manual_seed(1)

    steps = [(str(step.initial), str(step.rule)) for step in trace.all_steps]

    used_symbols = set([symbol for symbol, _ in steps])
    pre_symbol_to_ix = {s: i for i, s in enumerate(used_symbols)}

    # Used later for LSTM
    # idents = list(trace.meta.used_idents)
    # ident_to_ix = {ident: i for i, ident in enumerate(idents)}

    rules = [str(rule) for rule in trace.meta.rules]
    rule_to_ix = {rule: i for i, rule in enumerate(rules)}
    ix_to_rule = {i: rule for i, rule in enumerate(rules)}

    # Model
    model = IdentModeler(len(used_symbols), 32, len(rules))

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()

    losses = []
    for epoch in range(20):
        total_loss = 0
        for (symbol, rule) in steps:
            symbol = torch.tensor([pre_symbol_to_ix[symbol]], dtype=torch.long)

            model.zero_grad()

            log_probs = model(symbol)

            loss = loss_function(log_probs, torch.tensor(
                [rule_to_ix[rule]], dtype=torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)

    print(losses)

    # Trained predictions
    with torch.no_grad():
        OKGREEN = '\033[92m'
        ENDC = '\033[0m'
        for (symbol, rule) in steps[:4]:
            symbol_tensor = torch.tensor(
                [pre_symbol_to_ix[symbol]], dtype=torch.long)

            log_probs = model(symbol_tensor).flatten().tolist()

            # Find top three
            top_three = sorted(
                [(v, i) for i, v in enumerate(log_probs)], reverse=True)[:5]

            print(f'prev: {symbol}')
            print(f'expected: {rule}')

            for (v, i) in top_three:
                r = ix_to_rule[i]
                if r == rule:
                    print(f'{OKGREEN}{r} with {v}{ENDC}')
                else:
                    print(f'{r} with {v}')
            print('')
