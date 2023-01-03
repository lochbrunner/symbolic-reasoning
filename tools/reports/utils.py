from datetime import datetime
from .flavor import Flavors


def write_table(f, head, rows, think_h=False):
    f.write('\\begin{center}\n')
    align = '|'.join(['r' for _ in head])
    f.write(f'\\begin{{tabular}}{{{align}}}\n')
    head = ' & '.join(head).replace('_', ' ')
    f.write(f'{head} \\\\\n')
    if think_h:
        f.write('\\Xhline{2\\arrayrulewidth}')
    else:
        f.write('\\hline\n')
    for row in rows:
        row = ' & '.join([str(c) for c in row]).replace('_', ' ')
        f.write(f'{row} \\\\\n')
        if think_h:
            f.write('\\hline\n')

    f.write('\\end{tabular}\n')
    f.write('\\end{center}\n')


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


def roman_numeral(number):
    assert number < 1000
    row = [[], [0], [0, 0], [0, 0, 0], [0, 1], [1], [1, 0], [1, 0, 0], [1, 0, 0, 0], [0, 2]]
    char = [['I', 'V', 'X'],
            ['X', 'L', 'C'],
            ['C', 'D', 'M']]

    def digit(i, j):
        return ''.join([char[i][d] for d in row[j]])
    return digit(2, number//100)+digit(1, (number//10) % 10)+digit(0, number % 10)


def write_header(f):
    date = datetime.now().strftime('%a %b %d %Y')
    f.write('\\documentclass[usenames,dvipsnames]{scrartcl}\n')

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
        'makecell',
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
