
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
