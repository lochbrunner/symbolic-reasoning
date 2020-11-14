import logging
import sys


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


class Compose(object):
    '''Composes several transforms together.

    Transforms on a generic tuple instead of on value
    as torchvision.transforms.Compose does.
    '''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        for t in self.transforms:
            args = t(*args, **kwargs)
        return args


def setup_logging(verbose, log, **kwargs):
    loglevel = 'INFO' if verbose else log.upper()
    if sys.stdin.isatty():
        log_format = '%(message)s'
    else:
        log_format = '%(asctime)s %(message)s'

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format=log_format,
        datefmt='%I:%M:%S'
    )
    # Set the log level of all existing loggers
    all_loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for mod_logger in all_loggers:
        mod_logger._cache.clear()  # pylint: disable=protected-access
        mod_logger.setLevel(logging.getLevelName(loglevel))
