from pycore import Trace, Symbol, Rule
import sys


def load_trace():
    '''
    Loads the standard trace information
    '''
    file = '../out/generator/trace-1-1-1.bin'

    try:
        trace = Trace.load(file)
    except Exception as e:
        print(f'Error loading {file}: {e}')
        sys.exit(1)
    return trace


class Step:
    '''
    Simple struct holding relevant information a single step
    '''

    def __init__(self, initial, deduced, rule, path):
        self.initial = initial
        self.deduced = deduced
        self.rule = rule
        self.path = path
