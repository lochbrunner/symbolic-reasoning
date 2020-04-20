import subprocess as sp
import sys


def terminal_size():
    rows, columns = sp.check_output(['stty', 'size']).split()
    return int(rows), int(columns)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, fill='█', bg='-', percent=None, complete=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if sys.stdin.isatty():
        _, available_columns = terminal_size()
        length = available_columns - 10 - len(prefix) - len(suffix)

        percent = percent or 100 * (iteration / float(total))

        percent = ("{0:." + str(decimals) + "f}").format(percent)
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + bg * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        # Print New Line on Complete
        if complete and iteration == total:
            print()


def clearProgressBar():
    if sys.stdin.isatty():
        _, available_columns = terminal_size()
        print('\r' + ' ' * available_columns, end='\r')


def printHistogram(values, labels, total):
    if sys.stdin.isatty():
        s = sum(values)
        remaining = total - s
        m = max(values)
        m = max(m, remaining)
        longest_value = len(str(m))
        for v, l in zip(values, labels):
            label = str(l).rjust(2)
            value = str(v).rjust(longest_value)
            printProgressBar(v, m, prefix=label, suffix=value, percent=100*v/total, complete=False, bg=' ', fill='⣿')
            print()
        value = str(remaining).rjust(longest_value)
        label = '=>'.rjust(2)
        printProgressBar(remaining, m, prefix=label, suffix=value, percent=100 *
                         remaining/total, complete=False, bg=' ', fill='⣿')
        print()
