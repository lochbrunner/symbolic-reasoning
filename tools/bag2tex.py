#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
from pycore import Bag
import argparse
import math
import numpy as np
import scipy.optimize as optimize
import yaml

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt
import matplotlib
from flavor import Flavors
matplotlib.use('Agg')


def roman_numeral(number):
    assert number < 1000
    row = [[], [0], [0, 0], [0, 0, 0], [0, 1], [1], [1, 0], [1, 0, 0], [1, 0, 0, 0], [0, 2]]
    char = [['I', 'V', 'X'],
            ['X', 'L', 'C'],
            ['C', 'D', 'M']]

    def digit(i, j):
        return ''.join([char[i][d] for d in row[j]])
    return digit(2, number//100)+digit(1, (number//10) % 10)+digit(0, number % 10)


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


def write_key_value_table(f, obj):
    f.write('\\begin{center}\n')
    f.write('\\begin{tabular}{r|r}\n')
    f.write(f'Name & Value \\\\\n')
    f.write('\\hline\n')
    for k, v in obj.items():
        k = k.replace('_', ' ')
        f.write(f'{k} & {v} \\\\\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{center}\n')


def write_table(f, head, rows):
    f.write('\\begin{center}\n')
    align = '|'.join(['r' for _ in head])
    f.write(f'\\begin{{tabular}}{{{align}}}\n')
    head = ' & '.join(head).replace('_', ' ')
    f.write(f'{head} \\\\\n')
    f.write('\\hline\n')
    for row in rows:
        row = ' & '.join([str(c) for c in row]).replace('_', ' ')
        f.write(f'{row} \\\\\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{center}\n')


def training_statistics(f, scenario):
    with open(scenario['files']['training-statistics'], 'r') as sf:
        statistics = yaml.load(sf, Loader=yaml.FullLoader)

    f.write('\n\\section{Training Statistics}\n')

    if not statistics[0]['results']:
        f.write('No training statistics available')
        return

    write_table(f,
                head=['Set'] + list(statistics[0]['parameter'].keys()),
                rows=[[roman_numeral(i+1)] + list(statistic['parameter'].values()) for i, statistic in enumerate(statistics)])

    # Comparision of the parameter sets
    f.write('\n\\subsubsection{Exact (without padding)}\n')
    x = [roman_numeral(i+1) for i in range(len(statistics))]

    total = statistics[0]['results'][-1]['error']['exact-no-padding']['total']

    def y(i):
        return [s['results'][-1]['error']['exact-no-padding']['tops'][i]/total for s in statistics]

    def sy(i):
        return [sum(s['results'][-1]['error']['exact-no-padding']['tops'][:i])/total for s in statistics]

    ax = plt.gca()
    for i in range(10):
        ax.bar(x, y(i), bottom=sy(i))
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    plt.xticks(np.arange(len(x)), x)
    ax.set_xlabel('parameter set')
    ax.set_ylabel('match probability')
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')

    # The tops
    f.write('\n\\subsection{Tops}\n')
    for i, statistic in enumerate(statistics):
        f.write(f'\n\\subsubsection{{Parameterset {roman_numeral(i+1)}}}\n')
        hyper_parameter = statistic['parameter']

        f.write('Hyper Parameter\n')
        write_key_value_table(f, hyper_parameter)
        results = statistic['results']
        last_error = results[-1]['error']
        tops(f, last_error['exact-no-padding'], 'Exact matches (no padding)')

        tops_progress(f, 'exact-no-padding', 'Progress of exact Matches', results)

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


def create_parent_map(trace, mapping=None):
    '''Returns a map which maps child id to parent id'''
    if mapping is None:
        mapping = {}
    for child in trace['childs']:
        mapping[id(child)] = id(trace)
        create_parent_map(child, mapping)
    return mapping


def write_eval_on_training_data(f, results):
    f.write('\\subsection{Trainings Data}\n')
    trainings_traces = results['training-traces']

    x = [trace['beam-size'] for trace in trainings_traces]
    suc_color = 'tab:blue'
    success_rate = [trace['succeeded'] / trace['total']
                    for trace in trainings_traces]
    duration = [trace['mean-duration']*1000. for trace in trainings_traces]

    plt.title('Solving success on trainings data')
    ax = plt.gca()
    ax.plot(x, success_rate, color=suc_color)
    ax.set_yticklabels([f'{x:,.0%}' for x in ax.get_yticks()])
    ax.set_xlabel('beam size')
    ax.set_ylabel('successful solving', color=suc_color)
    ax.tick_params(axis='y', labelcolor=suc_color)

    dur_color = 'tab:orange'
    ax_duration = ax.twinx()
    ax_duration.set_ylabel('duration [ms]', color=dur_color)
    ax_duration.plot(x, duration, color=dur_color)
    ax_duration.tick_params(axis='y', labelcolor=dur_color)

    tikzplotlib.clean_figure()
    f.write(tikzplotlib.get_tikz_code(axis_width='15cm', axis_height='8cm'))
    plt.clf()
    f.write('\\\\')


def evaluation_results(f, scenario):
    f.write('\n\\section{Evaluation Results}\n')
    with open(scenario['files']['evaluation-results'], 'r') as sf:
        results = yaml.load(sf, Loader=yaml.FullLoader)

    write_eval_on_training_data(f, results)

    f.write('\\subsection{Problems}\n')
    beam_size = scenario['evaluation']['problems']['beam-size']
    f.write(f'Beam size: {beam_size}\\\n')

    for problem in results['problems']:
        problem = Namespace(problem)
        f.write(f'\n\\subsubsection{{{problem.name}}}\n')
        f.write('\\begin{align}\n')
        f.write(f'{problem.initial_latex}\n')
        f.write('\\end{align}\n')

        num_stages = depth_of_trace(problem.trace)
        angles = {}
        RX = 4.
        RY = 2.
        if num_stages >= 2:
            caption = f'Rose {problem.name}'
            f.write('''\\begin{figure}[!htb]
            \\centerline{
                %\\resizebox{12cm}{!}{
                \\begin{tikzpicture}[
                    scale=0.5,
                    level/.style={thick, <-},
                ]\n''')

            parent_map = create_parent_map(problem.trace)

            # From inner to outer
            for stage in range(num_stages):
                symbols_of_stage = symbols(problem.trace, stage)
                assert symbols_of_stage != []
                d = math.pi * 2. / len(symbols_of_stage)
                rx = RX*stage
                ry = RY*stage
                # Calculate angles offset
                if stage == 0:
                    offset = 0.
                else:
                    def dis(a, b):
                        pi = math.pi * 2.
                        r = math.fmod(a+pi-b, pi)
                        return min(r, pi-r)
                    parent_angles = [angles[parent_map[s.id]] for s in symbols_of_stage]

                    def y(o):
                        return sum([(dis(p, i*d+o))**2 for i, p in enumerate(parent_angles)])
                    result = optimize.minimize(y, 0.0)
                    offset = result.x[0].item()

                for i, symbol in enumerate(symbols_of_stage):
                    # Initial guess
                    a = d*float(i)+offset
                    x = rx*math.sin(a)
                    y = ry*math.cos(a)
                    angles[symbol.id] = a
                    f.write(f'\\node[] at ({x:.3f}cm,{y:.3f}cm) ({symbol.id}) {{${symbol.latex}$}};\n')

            for stage in range(num_stages):
                symbols_of_stage = symbols(problem.trace, stage)
                for symbol in symbols_of_stage:
                    for child_id in symbol.child_ids:
                        f.write(f'\\draw[level] ({child_id}) -- ({symbol.id});\n')

            f.write(f'''\\end{{tikzpicture}}
                }}
                %}}
            \\caption{{{caption}}}
            \\end{{figure}}''')

        if not problem.success:
            f.write('Could not been solved\\\\\n')

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
    wd = Path(scenario['files']['working-folder'])
    texfile = wd / f'{experiment_name}.tex'

    bag = Bag.load(bagfile)
    print(f'Writing file {texfile} ...')
    with texfile.open('w') as f:
        date = datetime.now().strftime('%a %b %d %Y')
        f.write('\\documentclass{scrartcl}\n')

        usepackages(f, [
            Package('T1', 'fontenc'),
            'lmodern',
            Package('english', 'babel'),
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

            samples = list(container.samples[: 100])

            samples.sort(key=lambda s: s.initial.density, reverse=True)

            for sample in samples:
                # sample.initial
                f.write('\\begin{align}\n')
                f.write(f'{sample.initial.latex_verbose}\n')
                f.write('\\end{align}\n')
                f.write(f'Density {sample.initial.density:.3f}')

        f.write('\\end{document}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('bag content')
    parser.add_argument('scenario_file')
    args = parser.parse_args()

    main(parser.parse_args())
