import os


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, fill='█'):
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
    _, available_columns = os.popen('stty size', 'r').read().split()
    available_columns = int(available_columns)
    length = available_columns - 10 - len(prefix) - len(suffix)

    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def clearProgressBar():
    _, available_columns = os.popen('stty size', 'r').read().split()
    available_columns = int(available_columns)
    print('\r' + ' ' * available_columns, end='\r')


def create_batches(samples, batch_size):
    return [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]
