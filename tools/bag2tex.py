#!/usr/bin/env python3

from pathlib import Path
from pycore import Bag
import argparse
import numpy as np
import yaml
import logging


from reports.utils import verbatim, write_header
from reports.generation import write_generation
from reports.evaluation import write_evaluation_results
from reports.training import write_training_statistics

# Graphs
import tikzplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


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
    with args.config_filename.open() as f:
        scenario = yaml.full_load(f)

    experiment_name = scenario['name']
    bagfile = scenario['files']['trainings-data']
    wd = Path(scenario['files']['working-folder'])
    texfile = wd / f'{experiment_name}.tex'

    bag = Bag.load(bagfile)
    logger.info(f'Writing file {texfile} ...')
    with texfile.open('w') as f:
        write_header(f)

        write_config(f, scenario)

        write_training_statistics(f, scenario)
        write_evaluation_results(f, scenario)
        write_generation(f, scenario, bag)

        f.write('\\end{document}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('bag content')
    parser.add_argument('config_filename', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(parser.parse_args())
