#!/usr/bin/env python3

from pathlib import Path
import logging
import argparse
from common.solving_problems import create_solving_problems
from common.creator_utils import collect


logger = logging.getLogger(__name__)


def main(args):

    me = Path(__file__).absolute()

    config_path = me.parent / 'config.yaml'
    args.operations_reservoir = ('-', '+', '*', '/', '^')
    collect(
        args,
        config_path=config_path,
        factories=[create_solving_problems],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Problems creator')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    parser.add_argument('--left-length', default=3, type=int)
    parser.add_argument('--right-length', default=3, type=int)
    parser.add_argument('--num-symbols', default=2, type=int)
    main(parser.parse_args())
