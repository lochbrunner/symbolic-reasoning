#!/usr/bin/env python3

import argparse
import re
import pathspec
import os
from pathlib import Path


def add_pyfiles(file_names):
    '''For each <dir>/__pycache__/<file>.cpython-37.pyc add <dir>/file.py'''

    pycache_regex = r"(\S+)/__pycache__/(\S+)\.(?:.+)\.pyc$"
    pyfiles = []
    for file_name in file_names:
        matches = re.findall(pycache_regex, file_name)
        if len(matches) > 0:
            directory, filestamp = matches[0]
            pyfiles.append(os.path.join(directory, filestamp + '.py'))
    file_names.update(pyfiles)
    return file_names


def remove_ignored_files(file_names):
    ignore_file = Path('.traceignore')
    if ignore_file.is_file():
        with ignore_file.open('r') as f:
            def pre(name):
                return name[1:] if name.startswith('/') else name

            spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
            file_names = (os.path.realpath(file) for file in file_names if not spec.match_file(pre(file)))
    return file_names


def remove_directories(file_names):
    directories_names = set()
    for file_name in file_names:
        while len(file_name) > 1:
            file_name = os.path.dirname(file_name)
            directories_names.add(file_name)

    return file_names - directories_names


def add_additional_files(args, file_names):
    if args.additional_files_list is not None:
        with args.additional_files_list.open('r') as f:
            lines = (line.strip() for line in f.readlines() if not line.startswith('//') and not line.startswith('#'))
            # If one of the lines is a parent directory then remove everything
            # inside that directory in the previous file names list
            obsolete_files = set()
            for filename in file_names:
                for line in lines:
                    if line.startswith(filename):
                        obsolete_files.add(filename)
                        break
            file_names.update(lines)
            file_names -= obsolete_files
    return file_names


def main(args):

    with open(args.strace_file, 'r') as f:
        regex = r"open(?:at)?\([^\"]*\"([^\"]+)\", O_RDONLY(?:(?!No such file or directory).)*$"
        content = f.read()
        matches = re.finditer(regex, content, re.MULTILINE)
    file_names = (match.groups()[0] for match in matches)

    file_names = remove_ignored_files(file_names)

    # Converting generator to set
    file_names = set(file_names)

    file_names = add_pyfiles(file_names)
    file_names = remove_directories(file_names)
    file_names = add_additional_files(args, file_names)

    file_names = sorted(file_names)

    if args.output_file is None:
        print('\n'.join(file_names))
    else:
        with args.output_file.open('w') as f:
            f.write('\n'.join(file_names) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('strace_file')
    parser.add_argument('-O', '--output-file', default=None, type=Path)
    parser.add_argument('-a', '--additional-files-list', default=None, type=Path)
    main(parser.parse_args())
