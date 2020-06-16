#!/usr/bin/env python

import argparse
import re
import pathspec
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('strace_file')
    parser.add_argument('-o', '--output-file', default=None)
    args = parser.parse_args()

    with open(args.strace_file, 'r') as f:
        regex = r"open(?:at)?\([^\"]*\"([^\"]+)\", O_RDONLY(?:(?!No such file or directory).)*$"
        content = f.read()
        matches = re.finditer(regex, content, re.MULTILINE)
    file_names = [match.groups()[0] for match in matches]

    with open('.traceignore', 'r') as f:
        def pre(name):
            if name.startswith('/'):
                return name[1:]
            else:
                return name
        spec = pathspec.PathSpec.from_lines('gitwildmatch', f)
        file_names = [os.path.realpath(file) for file in file_names if not spec.match_file(pre(file))]

    with open('additional_files.txt') as f:
        file_names += [line for line in f.readlines() if not line.startswith('//')]
    unique_file_names = set(file_names)

    # For each <dir>/__pycache__/<file>.cpython-37.pyc add <dir>/file.py
    pycache_regex = r"(\S+)/__pycache__/(\S+)\.(?:.+)\.pyc$"
    pyfiles = []
    for file_name in file_names:
        matches = re.findall(pycache_regex, file_name)
        if len(matches) > 0:
            directory, filestamp = matches[0]
            pyfiles.append(os.path.join(directory, filestamp + '.py'))
    unique_file_names.update(pyfiles)

    # Remove directories
    directories_names = set()
    for file_name in file_names:
        while len(file_name) > 1:
            file_name = os.path.dirname(file_name)
            directories_names.add(file_name)

    if args.output_file is None:
        print('\n'.join(unique_file_names))
    else:
        with open(args.output_file, 'w') as f:
            f.write('\n'.join(unique_file_names-directories_names) + '\n')
