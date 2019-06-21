from pycore import Trace, Bag
import sys
import os


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


def load_bag(filename='../out/generator/bag-1-1-1.bin'):
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


def clear_line():
    if in_ipynb():
        return
    WIDTH = int(os.popen('stty size', 'r').read().split()[1])
    print('\r' + ' '*WIDTH, end="\r")


class TerminalProgressBar:
    WIDTH = int(os.popen('stty size', 'r').read().split()[1])

    def __init__(self, min, max):
        self.min = min
        self.max = max

    @property
    def value(self):
        return 0.0

    @value.setter
    def value(self, value):
        width = (value-self.min)/(self.max-self.min) * self.WIDTH
        bar = '#'*int(width)
        print(f'\r{bar}', end='\r', flush=True)


def in_ipynb():
    try:
        cfg = get_ipython().config
        # if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
        #     return True
        # else:
        #     return False
        return True
    except NameError:
        return False


try:
    from ipywidgets import FloatProgress
    from IPython.display import display
except Exception as e:
    pass


class ProgressBar:
    '''Wrapper for terminal or notebook progress bar'''

    def __init__(self, min, max):
        if in_ipynb():
            self.bar = FloatProgress(min=min, max=max)
            display(self.bar)
        else:
            self.bar = TerminalProgressBar(min, max)

    def update(self, value):
        self.bar.value = value
