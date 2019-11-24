#!/usr/bin/env python3

import logging

import torch
import torch.optim as optim
from torch.utils import data
from torch import nn

from deep.dataset import PermutationDataset, Embedder, Padder, Uploader
from deep.model import TrivialTreeTaggerBatched

from common.timer import Timer
from common.utils import printProgressBar, clearProgressBar, Compose

# See
# * https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: data.DataLoader):
    # print(f'Samples: {len(dataloader)}')
    true = 0
    # We assume batchsize of 1
    assert dataloader.batch_size == 1
    for x, y, s in dataloader:
        x = model(x, s).view(-1)
        # print(f'x: {x.size()}')
        _, arg_max = x.max(0)
        if arg_max == y:
            true += 1
    return float(true) / float(len(dataloader))


def main():
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:0' if use_cuda else 'cpu')
    device = torch.device('cpu')  # pylint: disable=no-member

    logging.info(f'Using device: {device}')

    train_loader_params = {'batch_size': 16,
                           'shuffle': True,
                           'num_workers': 0}

    validate_loader_params = {'batch_size': 1,
                              'shuffle': False,
                              'num_workers': 0}

    timer = Timer('Loading samples')
    dataset = PermutationDataset(transform=Compose([
        Embedder(),
        Padder(),
        Uploader(device)
    ]))
    training_dataloader = data.DataLoader(dataset, **train_loader_params)
    validation_dataloader = data.DataLoader(dataset, **validate_loader_params)
    timer.stop_and_log()

    max_epochs = 30
    padding_index = 0

    model = TrivialTreeTaggerBatched(
        vocab_size=dataset.vocab_size, tagset_size=dataset.tag_size, pad_token=padding_index, hyper_parameter={'embedding_size': 32})

    timer = Timer('Sending model to device')
    model.to(device)
    timer.stop_and_log()

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    report_rate = 5
    timer = Timer('Training per interation')
    for epoch in range(max_epochs):
        epoch_loss = 0
        model.zero_grad()
        for x, y, s in training_dataloader:
            optimizer.zero_grad()
            x = model(x, s)

            # batch x tags
            loss = loss_function(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % report_rate == 0:
            error = validate(model, validation_dataloader)
            clearProgressBar()
            error = (1. - error) * 100.
            print(f'#{epoch} Loss: {epoch_loss:.3f}  Error: {error:.1f}')
        printProgressBar(epoch, max_epochs)
    clearProgressBar()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging._nameToLevel['INFO'],
        format='%(message)s'
    )
    main()
