#!/usr/bin/env python3

from pycore import Bag
from datetime import datetime
import yaml
import argparse
import os
import numpy as np
import math

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter, MultipleLocator
from flavor import Flavors
matplotlib.use('Agg')


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
    ax.barh(y_pos, counts, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('count')

    height = int(len(counts)/1.8)

    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='12cm', axis_height=f'{height}cm'))
    plt.clf()
    f.write('\\\\\n')
    dist = meta.rule_distribution[1:]
    f.write(f'Total number of usage: {sum(dist)} \\\\\n')
    f.write(f'Gini: {gini(dist):.2f}')


def tops(f, ratio, title):
    total = ratio['total']
    exact_tops = ratio['tops']
    exact_tops = [t/total for t in exact_tops]

    plt.title(title)
    ax = plt.gca()
    ax.bar(range(1, len(exact_tops)+1), exact_tops)
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')

    plt.title(f'{title} accumulated')
    cum_exact_tops = np.cumsum(exact_tops)
    ax = plt.gca()
    ax.bar(range(1, len(exact_tops)+1), cum_exact_tops)
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()

    f.write('\\\\')
    f.write(f'Remaining: {(1. - sum(exact_tops))*100:.2f}\\%')
    f.write('\\\\')


def tops_progress(f, key, title, statistics, count=8):
    total = statistics[0]['error'][key]['total']
    x = np.array([record['epoch'] for record in statistics])
    y = np.array([record['error'][key]['tops'][:count] for record in statistics], dtype=np.float32)

    y = np.transpose(y)
    y /= total

    plt.title(title)
    ax = plt.gca()
    ax.stackplot(x, y, labels=[f'top {(i+1)}' for i in range(y.shape[0])])
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    ax.legend(loc='lower right')
    ax.set_xlabel('epoch')

    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')


def training_statistics(f, scenario):
    with open(scenario['files']['training-statistics'], 'r') as sf:
        statistics = yaml.load(sf, Loader=yaml.FullLoader)

    f.write('\n\\section{Training Statistics}\n')
    f.write('\n\\subsection{Tops}\n')
    last_error = statistics[-1]['error']
    tops(f, last_error['exact-no-padding'], 'Exact matches (no padding)')

    tops_progress(f, 'exact-no-padding', 'Progress of exact Matches', statistics)

    # When rule
    when_rule_tops = last_error['when-rule']['tops']
    plt.bar(range(1, len(when_rule_tops)+1), when_rule_tops)
    plt.title('When rule')
    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')

    # When rule
    with_padding = last_error['with-padding']['tops']
    plt.bar(range(1, len(with_padding)+1), with_padding)
    plt.title('With padding')
    # tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()


class Namespace:
    def __init__(self, d):
        self.name = None
        self.initial_latex = None
        self.fit_results = None
        self.fit_tries = None
        self.success = None
        self.trace = None
        self.__dict__ = d


def depth_of_trace(trace):
    childs = trace['childs']
    if len(childs) == 0:
        return 1
    return max([depth_of_trace(c) for c in childs]) + 1


class Node:
    def __init__(self, latex, i, childs):
        self.latex = latex
        self.id = i
        self.child_ids = [id(c) for c in childs]


def symbols(trace, stage, current=0):
    if stage == current:
        return [Node(trace['apply_info']['current'], id(trace), trace['childs'])]

    return [s for c in trace['childs'] for s in symbols(c, stage, current+1)]


def evaluation_results(f, scenario):
    f.write('\n\\section{Evaluation Results}\n')

    f.write('\\subsection{Problems}\n')
    beam_size = scenario['evaluation']['beam-size']
    f.write(f'Beam size: {beam_size}\\\n')

    with open(scenario['files']['evaluation-results'], 'r') as sf:
        results = yaml.load(sf, Loader=yaml.FullLoader)

    for problem in results['problems']:
        problem = Namespace(problem)
        f.write(f'\n\\subsubsection{{{problem.name}}}\n')
        f.write('\\begin{align}\n')
        f.write(f'{problem.initial_latex}\n')
        f.write('\\end{align}\n')

        num_stages = depth_of_trace(problem.trace)
        angles = {}
        RX = 5.
        RY = 3.
        if num_stages >= 2:
            caption = f'Rose {problem.name}'
            f.write('''\\begin{figure}
            \\centerline{
                %\\resizebox{12cm}{!}{
                \\begin{tikzpicture}[
                    scale=0.5,
                    level/.style={thick, <-},
                ]\n''')

            for stage in reversed(range(num_stages)):
                symbols_of_stage = symbols(problem.trace, stage)
                assert symbols_of_stage != []

                # Which symbols are new?
                # Group leafs
                leafs = filter(lambda s: len(s.child_ids) == 0, symbols_of_stage)
                nodes = filter(lambda s: len(s.child_ids) > 0, symbols_of_stage)
                # Where are the childs?
                # First place nodes

                def mean(node):
                    childs = [angles[c] for c in node.child_ids]
                    return sum(childs) / len(childs)
                stage_angles = {n.id: mean(n) for n in nodes}
                # Place the leafs where is most place
                used_angles = [stage_angles[k] for k in stage_angles]
                used_angles.sort()

                def max_space(l):
                    if len(l) == 0:
                        # First node is right
                        return math.pi / 2.0
                    r = []
                    pi = math.pi * 2.
                    for i, _ in enumerate(l):
                        c = l[i]
                        p = l[i-1]

                        r.append(math.fmod(c-p+pi, pi))
                    max_index = r.index(max(r))
                    if l[max_index-1] < l[max_index]:
                        return (l[max_index] + l[max_index-1]) / 2.
                    else:
                        return math.fmod((l[max_index]+pi + l[max_index-1]) / 2., pi)

                for leaf in leafs:
                    angle = max_space(used_angles)
                    stage_angles[leaf.id] = angle
                    used_angles.append(angle)
                    used_angles.sort()

                for symbol in symbols_of_stage:
                    # a = d*i
                    a = stage_angles[symbol.id]
                    angles[symbol.id] = a
                    rx = RX*stage
                    ry = RY*stage
                    x = rx*math.sin(a)
                    y = ry*math.cos(a)

                    f.write(f'\\node[] at ({x:.3f}cm,{y:.3f}cm) ({symbol.id}) {{${symbol.latex}$}};\n')
                    for child_id in symbol.child_ids:
                        f.write(f'\\draw[level] ({child_id}) -- ({symbol.id});\n')

            f.write(f'''\\end{{tikzpicture}}
                }}
                %}}
            \\caption{{{caption}}}
            \\end{{figure}}''')

        if not problem.success:
            f.write('Could not been solved')
            continue
        f.write(f'Tried fits: {problem.fit_tries}\\\\\n')
        f.write(f'Successfull fits: {problem.fit_results}\\\n')

    f.write('\\pagebreak\n')


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
            Package('T1', 'fontenc'),
            'lmodern',
            Package('ngerman', 'babel'),
            'amsmath',
            ('pgfplots', 'pgfplotstable'),
            'colortbl',
            'xcolor',
            'color',
            'hyperref',
            'tikz',
            Package('hang,small,bf', 'caption')
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
\\tableofcontents
\\pagebreak\n''')

        write_config(f, scenario)

        training_statistics(f, scenario)
        evaluation_results(f, scenario)

        f.write('\\section{Trainings Data}\n')
        rule_usage(f, bag.meta)
        container_size(f, bag.samples)

        for container in bag.samples:
            if len(container.samples) == 0:
                continue
            f.write(f'\n\\subsection{{Container {container.max_depth}}}\n\n')
            f.write(f'Max depth {container.max_depth}\n')
            f.write(f'Max spread {container.max_spread}\n')
            f.write(f'Size {len(container.samples)}\n')

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
