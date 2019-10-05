import matplotlib.pyplot as plt
from os import path, makedirs
import pickle


class TrainingProgress:
    def __init__(self, iteration, loss, error):
        self.iteration = iteration
        self.loss = loss
        self.error = error


def plot_error(ax, progress, color, label='Error'):
    ax.set_xlabel('epoche')
    ax.set_ylabel(f'error [%]')
    ax.set_ylim(ymin=0, ymax=100)
    ax.plot([step.iteration for step in progress],
            [step.error*100.0 for step in progress], label=label, color=color)
    ax.tick_params(axis='y')


def plot_loss(ax, progress, color, label='Loss'):
    ax.set_ylabel('loss')
    ax.plot([step.iteration for step in progress],
            [step.loss for step in progress], label=label, color=color)
    ax.tick_params(axis='y')
    ax.set_ylim(ymin=0)


def plot_train_progess(progress, strategy, use, plot_filename='../reports/flat-training.{}.{}.svg', dump_filename='../reports/dump.p'):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    plot_error(ax1, progress, 'tab:red')

    ax2 = ax1.twinx()
    plot_loss(ax2, progress, 'tab:blue')

    plt.legend()

    fig.tight_layout()
    concret_plot_filename = plot_filename.format(strategy, use)
    print(f'Saving plot to {concret_plot_filename}...')
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


def draw_dump(plot_filename='../reports/summary/{}.{}.svg', dump_filename='../reports/dump.p'):
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
    }

    for strategy in dump:
        fig, ax = plt.subplots(figsize=(8, 6))
        for use in dump[strategy]:
            progress = dump[strategy][use]
            plot_error(ax, progress, color_table[use], label=use)
        plt.legend()
        filename = plot_filename.format(strategy, 'error')
        makedirs(path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

        fig, ax = plt.subplots(figsize=(8, 6))
        for use in dump[strategy]:
            progress = dump[strategy][use]
            plot_loss(ax, progress, color_table[use], label=use)
        plt.legend()
        filename = plot_filename.format(strategy, 'loss')
        makedirs(path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
