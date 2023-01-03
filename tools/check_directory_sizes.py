#!/usr/bin/env python3

from pathlib import Path
import argparse
from hurry.filesize import size
from pathspec import PathSpec
from dataclasses import dataclass


@dataclass
class Item:
    name: Path
    size: int


def main(args):

    with args.ignore_file.open() as f:
        spec = PathSpec.from_lines('gitwildmatch', f)

    def report(item: Item):
        if item.name.is_dir():
            d = 'd'
        else:
            d = ' '
        print(f'{str(item.name):<30}{d} {size(item.size)}')

    def calc_items():
        for item in args.root.iterdir():
            if item.is_file():
                if not spec.match_file(str(item)):
                    yield Item(item, item.stat().st_size)
            else:
                s = sum(f.stat().st_size for f in args.root.glob(
                    str(item / '**/*')) if f.is_file() and not spec.match_file(str(f)))
                if s > 0:
                    yield Item(item, s)

    items = list(calc_items())
    items.sort(key=lambda item: item.size, reverse=True)
    for item in items:
        report(item)
    print('-'*50)
    total = sum(item.size for item in items)
    report(Item(args.root, total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--ignore-file', type=Path, default='.amlignore')
    main(parser.parse_args())
