from io import IOBase
import numpy as np
import logging

from pycore import Bag, Scenario

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def density_histogram(f, container, scenario):
    f.write('\\\\')
    densities = [sample.initial.density for sample in container.samples]
    min_r = scenario['generation']['min-result-density']
    bins = int((1. - min_r) * 20)
    plt.hist(densities, bins=bins, range=(min_r, 1))
    plt.title('Density Distribution')
    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='6cm'))
    plt.clf()
    f.write('\\\\\n')


def gini(distribution):
    total = sum(distribution)
    nom = sum([sum([abs(xi-xj) for xj in distribution[i:]])
               for (i, xi) in enumerate(distribution)])
    return nom / (2.*len(distribution)*total)


def rule_usage(f, meta):
    f.write('\\subsection{Rule Distribution}\n')

    rules_tuple = list(zip(meta.rules[1:], meta.rule_distribution[1:]))
    rules_tuple.sort(key=lambda r: r[1], reverse=True)
    labels = [rule.name for (rule, _) in rules_tuple]
    counts = [count for (_, count) in rules_tuple]
    y_pos = np.arange(len(labels))

    plt.title('Rule usage')
    ax = plt.gca()
    positive, negative = zip(*counts)
    ax.barh(y_pos, positive, align='center', label='positive')
    ax.barh(y_pos, negative, align='center', label='negative')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('count')

    height = int(len(counts)/1.8)

    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='12cm', axis_height=f'{height}cm'))
    plt.clf()
    f.write('\\\\\n')
    dist = [p+n for p, n in meta.rule_distribution[1:]]
    f.write(f'Total number of usage: {sum(dist)} \\\\\n')
    f.write(f'Gini: {gini(dist):.2f}')


def container_size(f, containers):
    f.write('\\\\')
    sizes = [len(container.samples) for container in containers]

    plt.title('Depth Distribution')
    ax = plt.gca()
    ax.set_xlabel('depth')
    x_pos = np.arange(len(sizes))
    ax.bar(x_pos, sizes)

    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='6cm'))
    plt.clf()
    f.write('\\\\')
    f.write(f'Sum: {sum([len(c.samples) for c in containers])}')
    f.write('\\\\')


def write_generation(f: IOBase, scenario: Scenario, bag: Bag):
    logger.info('Generation')
    f.write('\\section{Trainings Data}\n')
    rule_usage(f, bag.meta)
    container_size(f, bag.containers)

    for container in bag.containers:
        if len(container.samples) == 0:
            continue
        f.write(f'\n\\subsection{{Container {container.max_depth}}}\n\n')
        f.write(f'Max depth {container.max_depth}\n')
        f.write(f'Max spread {container.max_spread}\n')
        f.write(f'Size {len(container.samples)}\n')

        if len(container.samples) > 0:
            density_histogram(f, container, scenario)

        samples = list(container.samples[: 100])

        samples.sort(key=lambda s: s.initial.density, reverse=True)

        for sample in samples:
            # sample.initial
            f.write('\\begin{align}\n')
            f.write(f'{sample.initial.latex_verbose}\n')
            f.write('\\end{align}\n')
            f.write(f'Density {sample.initial.density:.3f}')
