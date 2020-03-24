#!/usr/bin/env python3

from pycore import Bag
from datetime import datetime

import argparse


def main(args):
    bag = Bag.load(args.bagfile)
    with open(args.texfile, 'w') as f:
        date = datetime.now().strftime('%a %b %d %Y')
        f.write(f'''\\documentclass{{scrartcl}}

\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage[ngerman]{{babel}}
\\usepackage{{amsmath}}

\\usepackage{{xcolor}}

\\title{{Symbols}}
\\author{{Matthias Lochbrunner}}
\\date{{{date}}}
\\begin{{document}}

\\maketitle
\\tableofcontents
\\section{{Symbols}}
''')

        for i, container in enumerate(bag.samples):
            f.write(f'\n\\subsection{{Container {i}}}\n\n')
            for sample in container.samples:
                # sample.initial
                f.write('\\begin{align}\n')
                f.write(f'{sample.initial.latex}\n')
                f.write('\\end{align}\n')

        f.write('\end{document}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('bag content')
    parser.add_argument('bagfile')
    parser.add_argument('texfile')
    args = parser.parse_args()

    main(parser.parse_args())
