import yaml
import numpy as np
import logging

from .utils import roman_numeral, write_table

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
    f.write('Name & Value \\\\\n')
    f.write('\\hline\n')
    for k, v in obj.items():
        k = k.replace('_', ' ')
        f.write(f'{k} & {v} \\\\\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{center}\n')


def write_training_statistics(f, scenario):
    logger.info('Training statistics')
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
