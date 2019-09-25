from string import ascii_lowercase as alphabet
from itertools import permutations
from random import shuffle


def _create_permutation_samples(length=5):
    idents = alphabet[:length]
    classes = []
    samples = []

    for (i, perm) in enumerate(permutations(idents)):
        classes.append(i)
        samples.append((i, perm))

    return samples, list(idents), classes


def _create_haystack_needle(length=5, noise=2):
    idents = alphabet[1:length]

    classes = []
    samples = []

    indices = list(range(length))
    shuffle(indices)
    for index in indices:
        classes.append(index)
        for _ in range(noise):
            noise_array = list(idents)
            shuffle(noise_array)
            sample = noise_array[:index] + ['a'] + noise_array[index+1:]
            samples.append((index, sample))

    return samples, ['a'] + list(idents), classes


def _create_void_needle(length=5):
    idents = ['a', 'b']

    classes = []
    samples = []

    indices = list(range(length))
    shuffle(indices)

    for index in indices:
        classes.append(index)
        sample = ['b']*length
        sample[index] = 'a'
        samples.append((index, sample))

    return samples, idents, classes


def create_samples(strategy='permutation', **kwargs):
    if strategy == 'permutation':
        return _create_permutation_samples(**kwargs)
    elif strategy == 'haystack_needle':
        return _create_haystack_needle(**kwargs)
    elif strategy == 'void_needle':
        return _create_void_needle(**kwargs)
