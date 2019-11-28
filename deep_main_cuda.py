#!/usr/bin/env python3

import logging
import argparse

import torch
import torch.optim as optim
from torch.utils import data
from torch import nn

from deep.dataset import PermutationDataset, Embedder, Padder, Uploader, scenarios_choices, ScenarioParameter
from deep.models.trivial import TrivialTreeTagger

from common.timer import Timer
from common.utils import printProgressBar, clearProgressBar, Compose
from common.parameter_search import LearningParmeter

# See
# * https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class ExecutionParameter:
    def __init__(self, report_rate: int = 10, load_model: str = None, save_model: str = None):
        self.report_rate = report_rate
        self.load_model = load_model
        self.save_model = save_model


@torch.no_grad()
def validate(model: torch.nn.Module, dataloader: data.DataLoader):
    true = 0
    false = 0
    # We assume batchsize of 1
    assert dataloader.batch_size == 1
    for x, y, s in dataloader:
        x = model(x, s).view(-1)
        _, arg_max = x.max(0)
        if arg_max == y:
            true += 1
        else:
            false += 1
    return float(true) / float(true + false)


def main(exe_params: ExecutionParameter, learn_params: LearningParmeter, scenario_params: ScenarioParameter):
    # use_cuda = torch.cuda.is_available()
    # device = torch.device('cuda:0' if use_cuda else 'cpu')
    device = torch.device('cpu')  # pylint: disable=no-member

    logging.info(f'Using device: {device}')

    train_loader_params = {'batch_size': learn_params.batch_size,
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

    padding_index = 0

    model = TrivialTreeTagger(
        vocab_size=dataset.vocab_size,
        tagset_size=dataset.tag_size,
        pad_token=padding_index,
        hyper_parameter=learn_params.model_hyper_parameter)

    loss_function = nn.NLLLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=learn_params.learning_rate)

    if exe_params.load_model is not None:
        timer = Timer('Loading model from {exe_params.load_model}')
        checkpoint = torch.load(exe_params.load_model)
        file_use = checkpoint['use']
        current_use = 'default'
        if file_use != current_use:
            raise Exception(
                f'Loaded model contains {file_use} but {current_use} is specified for this training.')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        timer.stop_and_log()

    timer = Timer('Sending model to device')
    model.to(device)
    timer.stop_and_log()

    timer = Timer(f'Training per sample:')
    for epoch in range(learn_params.num_epochs):
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
        if epoch % exe_params.report_rate == 0:
            timer.pause()
            error = validate(model, validation_dataloader)
            clearProgressBar()
            error = (1. - error) * 100.
            loss = learn_params.batch_size * epoch_loss
            logging.info(f'#{epoch} Loss: {loss:.3f}  Error: {error:.1f}%')
            timer.resume()
        printProgressBar(epoch, learn_params.num_epochs)
    clearProgressBar()
    timer.stop_and_log_average(learn_params.num_epochs*len(dataset))

    if exe_params.save_model is not None:
        logging.info(f'Saving model to {exe_params.save_model} ...')
        torch.save({
            'epoch': learn_params.num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'use': 'default',
            'hyper_parameter': learn_params.model_hyper_parameter}, exe_params.save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    # Execution parameter
    parser.add_argument('-i', '--load-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-o', '--save-model', type=str,
                        default=None, help='Path to the model')
    parser.add_argument('-r', '--report-rate', type=int, default=20)

    # Learning parameter
    parser.add_argument('-n', '--num-epochs', type=int, default=30)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.1)
    parser.add_argument('-c', '--gradient-clipping', type=float, default=0.1)
    # Scenario
    parser.add_argument('-s', '--scenario', type=str,
                        default='permutation', choices=scenarios_choices() + ['all'])
    parser.add_argument('--depth', type=int, default=2,
                        help='The depth of the used nodes.')
    parser.add_argument('--spread', type=int, default=2)

    args = parser.parse_args()
    loglevel = 'INFO' if args.verbose else args.log.upper()

    logging.basicConfig(
        level=logging._nameToLevel[loglevel],
        format='%(message)s'
    )

    exec_params = ExecutionParameter(
        report_rate=args.report_rate, load_model=args.load_model,
        save_model=args.save_model)

    learn_params = LearningParmeter(
        num_epochs=args.num_epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, gradient_clipping=args.gradient_clipping,
        model_hyper_parameter={})

    scenario_params = ScenarioParameter(
        scenario=args.scenario, depth=args.depth, spread=args.spread)

    main(exec_params, learn_params, scenario_params=scenario_params)
