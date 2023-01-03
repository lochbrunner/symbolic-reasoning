from os import path, makedirs
from typing import Dict
import logging
import matplotlib.pyplot as plt
import pickle

from solver.metrics import Tops

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = type(None)


def report_tops(
    tops: Tops, epoch: int, writer: SummaryWriter = None, label='tops'
) -> None:
    total = float(tops.total)

    N = 7

    def top_k_str(i):
        return f'#{i}: {top_k(i):.3f}'

    def top_k(i):
        if i in tops.values:
            return tops.values[i] / total
        return 0.0

    if total > 0:
        rest = sum(v for i, v in tops.values.items() if i >= N) / total
    else:
        rest = 0

    if writer:
        scalars = {f'top_{i}': top_k(i) for i in range(1, N)}
        scalars['rest'] = rest
        writer.add_scalars(label, scalars, epoch)
        writer.add_scalar(f'{label}/worst', float(tops.worst), epoch)
        writer.flush()

    if not writer:
        tops_str = ', '.join(top_k_str(i) for i in range(1, N))
        print(f'{label}: {tops_str}, rest: {rest:.3f}, worst: {tops.worst:.3f}')


class TrainingProgress:
    def __init__(self, iteration, loss, error):
        self.iteration = iteration
        self.loss = loss
        self.error = error


def plot_error(ax, progress, color, label='Error'):
    ax.set_xlabel('epoche')
    ax.set_ylabel(f'error [%]')
    ax.set_ylim(ymin=0, ymax=100)
    ax.plot(
        [step.iteration for step in progress],
        [step.error * 100.0 for step in progress],
        label=label,
        color=color,
    )
    ax.tick_params(axis='y')


def plot_loss(ax, progress, color, label='Loss'):
    ax.set_xlabel('epoche')
    ax.set_ylabel('loss')
    ax.plot(
        [step.iteration for step in progress],
        [step.loss for step in progress],
        label=label,
        color=color,
    )
    ax.tick_params(axis='y')
    ax.set_ylim(ymin=0)


def plot_train_progess(
    progress,
    strategy,
    use,
    plot_filename='./reports/flat/training.{}.{}.svg',
    dump_filename='./reports/flat/dump.p',
):
    fig, ax1 = plt.subplots(figsize=(12, 9))
    plot_error(ax1, progress, 'tab:red')

    ax2 = ax1.twinx()
    plot_loss(ax2, progress, 'tab:blue')

    plt.legend()

    fig.tight_layout()
    concret_plot_filename = plot_filename.format(strategy, use)
    logging.info(f'Saving plot to {concret_plot_filename} ...')
    makedirs(path.dirname(concret_plot_filename), exist_ok=True)
    plt.savefig(concret_plot_filename)
    if path.isfile(dump_filename):
        with open(dump_filename, 'rb') as pickle_file:
            dump = pickle.load(pickle_file)
    else:
        dump = {}
    if strategy not in dump:
        dump[strategy] = {}
    dump[strategy][use] = progress
    pickle.dump(dump, open(dump_filename, 'wb'))


def draw_dump(
    plot_filename='./reports/summary/flat/{}.{}.svg', dump_filename='../reports/dump.p'
):
    with open(dump_filename, 'rb') as pickle_file:
        dump = pickle.load(pickle_file)

    # From https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    color_table = {
        'own': 'tab:blue',
        'torch-cell': 'tab:orange',
        'optimized': 'tab:green',
        'rebuilt': 'tab:red',
        'torch': 'tab:purple',
        'None': 'tab:purple',
        'optimized-two': 'tab:cyan',
    }

    for strategy in dump:
        fig, ax = plt.subplots(figsize=(12, 9))
        for use in dump[strategy]:
            progress = dump[strategy][use]
            plot_error(ax, progress, color_table[use], label=use)
        plt.legend()
        filename = plot_filename.format(strategy, 'error')
        makedirs(path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        logging.info(f'Writing file {filename}')

        fig, ax = plt.subplots(figsize=(12, 9))
        for use in dump[strategy]:
            progress = dump[strategy][use]
            plot_loss(ax, progress, color_table[use], label=use)
        plt.legend()
        filename = plot_filename.format(strategy, 'loss')
        makedirs(path.dirname(filename), exist_ok=True)
        logging.info(f'Writing file {filename}')
        plt.savefig(filename)
