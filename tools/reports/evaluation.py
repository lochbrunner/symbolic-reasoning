from .utils import write_table
from dataclasses import dataclass
from typing import Dict, List
import math
import scipy.optimize as optimize
import yaml
import logging

# Graph
import tikzplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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


def create_parent_map(trace, mapping=None):
    '''Returns a map which maps child id to parent id'''
    if mapping is None:
        mapping = {}
    for child in trace['childs']:
        mapping[id(child)] = id(trace)
        create_parent_map(child, mapping)
    return mapping


class Namespace:
    def __init__(self, d):
        self.name = None
        self.initial_latex = None
        self.fit_results = None
        self.fit_tries = None
        self.success = None
        self.trace = None
        self.__dict__ = d


@dataclass
class Node:
    latex: str
    id: int
    # child_ids
    childs: List
    contributed: bool
    value: float


def symbols(trace, stage, current=0):
    if stage == current:
        apply_info = trace['apply_info']
        return [Node(apply_info['current'],
                     id(trace),
                     trace['childs'],
                     contributed=apply_info['contributed'],
                     value=apply_info['value'])]

    return [s for c in trace['childs'] for s in symbols(c, stage, current+1)]


def depth_of_trace(trace):
    childs = trace['childs']
    if len(childs) == 0:
        return 1
    return max([depth_of_trace(c) for c in childs]) + 1


@ dataclass
class Step:
    current: Dict
    childs: List[Dict]


class ApplyInfo:
    def __init__(self, d):
        self.confidence: float = None
        self.contributed: bool = None
        self.current: str = None
        self.rule_name: str = None
        self.rule_id: int = None
        self.top: int = None
        self.value: float = None
        self.__dict__ = {k.replace('-', '_'): v for k, v in d.items()}


def reconstruct_solution_trace(trace):
    node = trace
    if not node['apply_info']['contributed']:
        yield from()

    try:
        while True:
            yield Step(ApplyInfo(node['apply_info']), [ApplyInfo(n['apply_info']) for n in node['childs']])
            node = next(n for n in node['childs'] if n['apply_info']['contributed'])

    except StopIteration:
        pass  # End of trace


def reconstruct_solution(trace):
    node = trace
    if not node['apply_info']['contributed']:
        yield from()
    try:
        while True:
            yield node['apply_info']
            node = next(n for n in node['childs'] if n['apply_info']['contributed'])

    except StopIteration:
        pass  # End of trace


def write_evaluation_results(f, scenario):
    logger.info('Evaluation results')
    f.write('\n\\section{Evaluation Results}\n')
    with open(scenario['files']['evaluation-results'], 'r') as sf:
        results = yaml.load(sf, Loader=yaml.FullLoader)

    write_eval_on_training_data(f, results)

    f.write('\\subsection{Problems}\n')
    beam_size = scenario['evaluation']['problems']['beam-size']
    f.write(f'Beam size: {beam_size}\\\n')

    RX = 4.
    RY = 2.5

    for problem in results['problems']:
        problem = Namespace(problem)
        f.write(f'\n\\subsubsection{{{problem.name}}}\n')
        f.write('\\begin{align}\n')
        f.write(f'{problem.initial_latex}\n')
        f.write('\\end{align}\n')

        num_stages = depth_of_trace(problem.trace)
        angles = {}
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

            MAX_STAGES = 10

            # From inner to outer
            for stage in range(min(num_stages, MAX_STAGES)):
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
                    value = int((symbol.value or 1)*100)
                    color = 'blue' if symbol.contributed else 'black'
                    color = f'{color}!{value}!white'
                    f.write(f'\\node[{color}] at ({x:.3f}cm,{y:.3f}cm) ({symbol.id}) {{\\small ${symbol.latex}$}};\n')

            for stage in range(min(num_stages, MAX_STAGES-1)):
                symbols_of_stage = symbols(problem.trace, stage)
                for symbol in symbols_of_stage:
                    for child in symbol.childs:
                        child_id = id(child)
                        value = int((child['apply_info']['value'] or 1)*100)
                        color = ',blue' if child['apply_info']['contributed'] else ',black'
                        color = f'{color}!{value}!white'
                        f.write(f'\\draw[level{color}] ({child_id}) -- ({symbol.id});\n')

            f.write(f'''\\end{{tikzpicture}}
                }}
                %}}
\\caption{{{caption}}}
\\end{{figure}}''')

        if not problem.success:
            f.write('Could not been solved\\\\\n')
        else:
            # Trace each step with sorted by policy and colored with their value

            def embed_formulas(terms):
                terms = list(terms)
                terms.sort(key=lambda term: term.top)

                def make_cell(term):
                    if term.contributed:
                        color = '\\color{MidnightBlue}'
                    else:
                        color = ''
                    if term.value:
                        return f'\\makecell{{{color}\\footnotesize ${term.current}$ \\\\ {term.value:.3f} }}'
                    else:
                        return f'{{{color}\\footnotesize ${term.current}$}}'

                return [make_cell(term) for term in terms]

            rows = [(
                f'${step.current.current}$',
                *embed_formulas(step.childs)
            )
                for step in reconstruct_solution_trace(problem.trace)]

            largest_row = max(len(row) for row in rows)-1

            head = ('initial', *(['child']*largest_row))
            write_table(f, head, rows, think_h=True)

            min_value = min(f['value'] for f in reconstruct_solution(problem.trace) if f['value'] is not None)
            f.write(f'Minimum value: ${min_value:.3f}$\\\\\n')

            max_top = max(f['top'] for f in reconstruct_solution(problem.trace) if f['top'] is not None)
            f.write(f'Maximum top: ${max_top}$\\\\\n')

        f.write(f'Tried fits: {problem.fit_tries}\\\\\n')
        f.write(f'Successfull fits: {problem.fit_results}\\\n')

    f.write('\\pagebreak\n')
