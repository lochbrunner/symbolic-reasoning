#!/usr/bin/env python3

import solving_problems
from common.creator_utils import collect, Replacer
from pathlib import Path
import logging
import argparse

from pycore import Symbol


logger = logging.getLogger(__name__)


def main(args):

    me = Path(__file__).absolute()

    config_path = me.parent / 'config.yaml'
    collect(
        args,
        config_path=config_path,
        factories=[
            solving_problems.create_quadratic_equations,
            solving_problems.create_exponential,
        ],
        replacer=Replacer("sqrt", "root", Symbol.number(2))
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Problems creator')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--left-length', default=3, type=int)
    parser.add_argument('--right-length', default=3, type=int)
    parser.add_argument('--num-symbols', default=2, type=int)
    main(parser.parse_args())
