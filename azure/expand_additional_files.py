#!/usr/bin/env python3

from glob import glob
import argparse
from pathlib import Path


def main(args):
    additional_files = set()

    with args.input_list_file.open('r') as f:
        for pattern in [line.strip() for line in f.readlines() if not line.startswith('#') and not line.startswith('//')]:
            additional_files.update(glob(pattern))

    # Remove directories
    directories_names = set()
    for file_name in additional_files:
        while len(file_name) > 1:
            file_name = Path(file_name)
            directories_names.add(file_name.parents[0])

    with args.output_list_file.open('w') as f:
        f.write('\n'.join(additional_files-directories_names) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-list-file', type=Path)
    parser.add_argument('-O', '--output-list-file', type=Path)
    main(parser.parse_args())
