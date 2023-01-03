#!/usr/bin/env python3

import requests
import argparse
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class Sample:
    timestamp: int
    step: int
    data: List[float]


def fetchSamples(args):
    response = [requests.get(
        f'http://localhost:{args.port}/data/plugin/scalars/scalars?tag=policy%2Fexact&run={args.scalar}+%5B{i}%5D').json()
        for i in range(10)]

    return [Sample(timestamp=data[0][0], step=data[0][1], data=[d[2] for d in data]) for data in
            zip(*response)
            ]


def writeLaTeXCode(args, samples: List[Sample]):
    nl = '\n'
    step_min = samples[0].step
    step_max = samples[-1].step
    with args.output.open('w') as f:
        def writeln(text: str = ''):
            f.write(text + nl)
        writeln(r'\begin{tikzpicture}')
        writeln(r'\begin{axis}[')
        writeln('   width=16cm,')
        writeln('   height=8cm,')
        writeln(r'   legend style={nodes={scale=0.5, transform shape}},')
        writeln(f'   title={{{args.title}}},')
        writeln(r'   ylabel={error [\%]},')
        writeln(f'   ytick={{{", ".join(str(x*10)for x in range(0,11))}}},')
        writeln(r'   xlabel={training step},')
        writeln(f'   xmin={step_min}, xmax={step_max},')
        writeln(f'   ymin={0}, ymax={100},')
        writeln(r'   legend pos=north east,')
        writeln(r'   ymajorgrids=true,')
        writeln(r'   grid style=dashed,')
        writeln(r']')
        writeln()

        for i in range(10):
            writeln(r'\addplot[')
            writeln(f'  color=blue!{i*10}!red,')
            writeln(r']')
            writeln(r'  coordinates {')
            f.write('  ')
            for sample in samples:
                f.write(f'({sample.step}, {sample.data[i]*100})')

            writeln()
            writeln(r'  };')
            writeln(f'\\addlegendentry{{top [{i+1}]}};')
        writeln(r'\end{axis}')
        writeln(r'\end{tikzpicture}')


def main(args):
    samples = fetchSamples(args)
    writeLaTeXCode(args, samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=6006, type=int)
    parser.add_argument('--scalar', default='policy_exact_top')
    parser.add_argument('--title', default='Exact policy match')
    parser.add_argument('-O', '--output', default='docs/whitepaper/training.tex', type=Path)
    main(parser.parse_args())
