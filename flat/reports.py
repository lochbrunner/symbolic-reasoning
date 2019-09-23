import matplotlib.pyplot as plt


class TrainingProgress:
    def __init__(self, iteration, loss, error):
        self.iteration = iteration
        self.loss = loss
        self.error = error


def plot_train_progess_dep(progress, filename='../reports/flat-training.svg'):
    fig = plt.figure(figsize=(8, 6))

    plt.plot([step.iteration for step in progress],
             [step.error for step in progress], label='Error')
    plt.plot([step.iteration for step in progress],
             [step.loss for step in progress], label='Loss')

    plt.legend()
    plt.savefig(filename)


def plot_train_progess(progress, filename='../reports/flat-training.svg'):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:red'
    ax1.set_xlabel('epoche')
    ax1.set_ylabel('error [%]', color=color)
    ax1.set_ylim(ymin=0, ymax=100)
    ax1.plot([step.iteration for step in progress],
             [step.error*100.0 for step in progress], label='Error', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # plt.legend()

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)
    ax2.plot([step.iteration for step in progress],
             [step.loss for step in progress], label='Loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(ymin=0)
    plt.legend()

    fig.tight_layout()
    plt.savefig(filename)
