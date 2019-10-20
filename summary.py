#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import argcomplete

from common.reports import draw_dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Summary drawing')
    parser.add_argument('-d', '--dump-filename', type=str)
    parser.add_argument('-p', '--plot-filename', type=str,
                        default='./reports/summary/flat/{}.{}.svg')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    draw_dump(plot_filename=args.plot_filename,
              dump_filename=args.dump_filename)
