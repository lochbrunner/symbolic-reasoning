#!/bin/sh

ROOT_DIR=dirname($0)

PYTHONPATH=$ROOT_DIR deep/generate.specs.py
PYTHONPATH=$ROOT_DIR common/parameter_search.specs.py
