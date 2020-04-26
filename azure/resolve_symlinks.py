#!/usr/bin/env python

import argparse
from pathlib import Path


def resolve(orig):
    handler = Path(orig)
    if handler.is_symlink():
        target = handler.resolve()
        return [str(target)] + resolve(target)
    else:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('raw_filelist')
    parser.add_argument('-o', '--output-file', default=None)
    args = parser.parse_args()

    with open(args.raw_filelist, 'r') as f:
        raw_filenames = f.readlines()

    additional_files = []
    for raw_filename in raw_filenames:
        additional_files += resolve(raw_filename.strip())

    additional_files = set(additional_files)
    # Remove intersection
    additional_files -= additional_files.intersection(raw_filenames)

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(additional_files) + '\n')
