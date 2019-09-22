from string import ascii_lowercase as alphabet
from itertools import permutations


def create_samples(size, strategy='permutation'):

    idents = alphabet[:size]
    classes = []
    samples = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)
        samples.append((i, perm))

    return samples, list(idents), classes
