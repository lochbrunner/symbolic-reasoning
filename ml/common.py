from pycore import Trace, Bag
import sys


def load_trace(filename='../out/generator/trace-1-1-1.bin'):
    '''
    Loads the standard trace information
    '''

    try:
        trace = Trace.load(filename)
    except Exception as e:
        print(f'Error loading {filename}: {e}')
        sys.exit(1)
    return trace


def load_bag(filename='../out/generator/bag-1-1.bin'):
    '''
    Loads the bag file
    '''

    try:
        bag = Bag.load(filename)
    except Exception as e:
        print(f'Error loading {filename}: {e}')
        sys.exit(1)
    return bag


def sanitize_path(path):
    return path.replace('/', '#')


class Step:
    '''
    Simple struct holding relevant information a single step
    '''

    def __init__(self, initial, deduced, rule, path):
        self.initial = initial
        self.deduced = deduced
        self.rule = rule
        self.path = path
