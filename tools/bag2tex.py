#!/usr/bin/env python3

from pycore import Bag
from datetime import datetime
import yaml
import argparse
import os
import numpy as np

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter
from flavor import Flavors
matplotlib.use('Agg')


def density_histogram(f, container, scenario):
    f.write('\\\\')
    densities = [sample.initial.density for sample in container.samples]
    min_r = scenario['generation']['min-result-density']
    bins = int((1. - min_r) * 20)
    plt.hist(densities, bins=bins, range=(min_r, 1))
    plt.title('Density Distribution')
    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='6cm'))
    plt.clf()
    f.write('\\\\\n')


def tops(f, ratio, title):
    total = ratio['total']
    exact_tops = ratio['tops']
    exact_tops = [t/total for t in exact_tops]

    plt.title(title)
    ax = plt.gca()
    ax.bar(range(1, len(exact_tops)+1), exact_tops)
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')

    plt.title(f'{title} accumulated')
    cum_exact_tops = np.cumsum(exact_tops)
    ax = plt.gca()
    ax.bar(range(1, len(exact_tops)+1), cum_exact_tops)
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()

    f.write('\\\\')
    f.write(f'Remaining: {(1. - sum(exact_tops))*100:.2f}\\%')
    f.write('\\\\')


def training_statistics(f, scenario):
    with open(scenario['files']['training-statistics'], 'r') as sf:
        statistics = yaml.load(sf, Loader=yaml.FullLoader)

    f.write('\n\\section{Training Statistics}\n')
    f.write('\n\\subsection{Tops}\n')
    # tops(f, statistics['exact'], 'Exact matches')
    tops(f, statistics['exact-no-padding'], 'Exact matches (no padding)')

    # When rule
    when_rule_tops = statistics['when-rule']['tops']
    plt.bar(range(1, len(when_rule_tops)+1), when_rule_tops)
    plt.title('When rule')
    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')

    # When rule
    with_padding = statistics['with-padding']['tops']
    plt.bar(range(1, len(with_padding)+1), with_padding)
    plt.title('With padding')
    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()


class Package:
    def __init__(self, o, p):
        self.o = o
        self.p = p


def usepackages(f, packages):
    for package in packages:
        if type(package) is str:
            f.write(f'\\usepackage{{{package}}}\n')
        elif type(package) is list or type(package) is tuple:
            arg = ', '.join(package)
            f.write(f'\\usepackage{{{arg}}}\n')
        elif type(package) is Package:
            o = package.o
            p = package.p
            f.write(f'\\usepackage[{o}]{{{p}}}\n')


def verbatim(code):
    if type(code) is not str:
        return str(code)
    c = ''
    c += '\\begin{minipage}{0.7in}\n'
    c += '\\begin{verbatim}\n'
    c += str(code)
    c += '\n\\end{verbatim}\n'
    c += '\\end{minipage}\n'

    return c


def write_config(f, scenario):
    generation = scenario['generation']
    f.write('''\\section{Configuration}
\\begin{tabular}{r r}\n''')
    for key in generation:
        if key in ['blacklist-pattern']:
            continue
        value = generation[key]
        if type(value) in [str, int, float]:
            f.write(f'{key} & {value} \\\\\n')
        elif type(value) is list:
            value = ', '.join([verbatim(v) for v in value])
            f.write(f'{key} & {value} \\\\\n')

    f.write('''\\end{tabular}\n''')


def main(args):
    with open(args.scenario_file, 'r') as f:
        scenario = yaml.load(f, Loader=yaml.FullLoader)

    experiment_name = scenario['name']
    bagfile = scenario['files']['trainings-data']
    texfile = os.path.join(scenario['files']['working-folder'], f'{experiment_name}.tex')

    bag = Bag.load(bagfile)
    print(f'Writing file {texfile} ...')
    with open(texfile, 'w') as f:
        date = datetime.now().strftime('%a %b %d %Y')
        f.write('\\documentclass{scrartcl}\n')

        usepackages(f, [
            Package('utf8', 'inputenc'),
            Package('T1', 'fontenc'),
            'lmodern',
            Package('ngerman', 'babel'),
            'amsmath',
            ('pgfplots', 'pgfplotstable'),
            'colortbl',
            'xcolor',
            'color',
            'hyperref'
        ])
        f.write(Flavors.latex.preamble())

        f.write(f'''
\\title{{Report}}
\\author{{Matthias Lochbrunner}}
\\date{{{date}}}
\\begin{{document}}

\\maketitle
\\hypersetup{{
    linktocpage,
    linkcolor = false,
    colorlinks = false
}}
\\tableofcontents\n''')

        write_config(f, scenario)

        training_statistics(f, scenario)

        f.write('\\section{Trainings Data}\n')

        for container in bag.samples:
            f.write(f'\n\\subsection{{Container {container.max_depth}}}\n\n')
            f.write(f'Max depth {container.max_depth}\n')
            f.write(f'Max spread {container.max_spread}\n')

            if len(container.samples) > 0:
                density_histogram(f, container, scenario)

            samples = list(container.samples)

            samples.sort(key=lambda s: s.initial.density, reverse=True)

            for sample in samples:
                # sample.initial
                f.write('\\begin{align}\n')
                f.write(f'{sample.initial.latex_verbose}\n')
                f.write('\\end{align}\n')
                f.write(f'Density {sample.initial.density:.3f}')

        f.write('\end{document}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('bag content')
    parser.add_argument('scenario_file')
    args = parser.parse_args()

    main(parser.parse_args())
