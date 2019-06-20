
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from common import sanitize_path


def highlight_cell(cells, ax=None, **kwargs):
    ax = ax or plt.gca()
    for (x, y) in cells:
        rect = plt.Rectangle((x-.5, y-0.5), 1, 1, fill=False, **kwargs)
        ax.add_patch(rect)
    return rect


def show_part_predictions(predict_rule, samples, ix_to_rule, rule_to_ix):
    fig = plt.figure(figsize=(8, 6))
    graph = fig.add_subplot(111)
    ax = plt.gca()

    size = len(samples)

    class Index:
        def __init__(self):
            self.index = 0
            self.cb = None
            self.draw()

        def next(self, event):
            self.index = (self.index + 1) % size
            self.draw()

        def prev(self, event):
            self.index = (self.index - 1) % size
            self.draw()

        def draw(self):
            graph.clear()
            self.cb = plot_part_prediction_impl(
                predict_rule, samples[self.index], ix_to_rule, rule_to_ix, graph, fig, ax=ax, cb=self.cb)
            plt.draw()

    index = Index()

    # Buttons
    axprev = plt.axes([0.7, 0.03, 0.1, 0.04])
    axnext = plt.axes([0.81, 0.03, 0.1, 0.04])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(index.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(index.prev)
    plt.show()


def plot_part_prediction(predict_rule, sample, ix_to_rule, rule_to_ix, filename_prefix='../out/ml/lstm-single-prediction'):
    fig = plt.figure(figsize=(8, 6))
    graph = fig.add_subplot(111)

    plot_part_prediction_impl(
        predict_rule, sample, ix_to_rule, rule_to_ix, graph, fig)

    plt.savefig(
        f'{filename_prefix}-{sanitize_path(str(sample.initial))}.svg')
    plt.show()


def plot_part_prediction_impl(predict_rule, sample, ix_to_rule, rule_to_ix, graph, fig, ax=None, cb=None):
    rules = [f'$ {r.latex} $' for i, r in ix_to_rule.items()]
    rules_ix = [i for i, r in ix_to_rule.items()]

    y_ticks = []
    prob_matrix = []
    path_to_ix = {}
    for i, (path, part) in enumerate(sample.initial.parts_with_path):
        path_to_ix[str(path)] = i
        log_probs, _ = predict_rule(part)
        probs = log_probs.flatten().tolist()
        prob_matrix.append(probs)
        y_ticks.append(f'$ {part.latex} $')
    im = graph.imshow(prob_matrix, cmap='Reds', interpolation='nearest')
    if ax is None:
        plt.xticks(rules_ix, rules, rotation=45)
        plt.yticks(list(range(len(y_ticks))), y_ticks)
    else:
        ax.set_xticks(rules_ix)
        ax.set_xticklabels(rules, rotation=45)
        ax.set_yticks(list(range(len(y_ticks))))
        ax.set_yticklabels(y_ticks)

    if cb is None:
        cb = fig.colorbar(im, ax=ax)

    for fit in sample.fits:
        rule_ix = rule_to_ix[fit.rule.verbose]
        ix = path_to_ix[str(fit.path)]
        highlight_cell([(rule_ix, ix)], color="limegreen",
                       linewidth=3, ax=ax)
    return cb
