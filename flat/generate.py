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


def _create_haystack_needle(length=8, noise=2):
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


def _create_void_needle(length=20):
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


def _count_pattern(sequence, pattern):
    findings = 0
    plen = len(pattern)
    for i in range(len(sequence) - plen+1):
        if sequence[i:i+plen] == pattern:
            findings += 1
    return findings


def _create_complex_pattern_in_noise(length=14, size=50, pattern_length=4):
    '''Embeds a unique and fixed pattern (beginning of the alphabet) into noise'''
    length += pattern_length
    pattern = list(alphabet[:pattern_length])
    idents = list(alphabet[:length])

    classes = list(range(0, length-pattern_length))
    samples = []
    while size > 0:
        index = size % len(classes)
        shuffle(idents)
        # Embed the pattern
        hypo = idents[:index] + pattern + idents[index+pattern_length:]
        # count patterns
        if _count_pattern(hypo, pattern) > 1:
            continue
        else:
            samples.append((index, hypo))
            size -= 1

    return samples, list(alphabet[:length]), classes


def choices():
    return ['permutation', 'haystack_needle', 'void_needle', 'pattern']


def create_samples(strategy='permutation', **kwargs):
    if strategy == 'permutation':
        return _create_permutation_samples(**kwargs)
    elif strategy == 'haystack_needle':
        return _create_haystack_needle(**kwargs)
    elif strategy == 'void_needle':
        return _create_void_needle(**kwargs)
    elif strategy == 'pattern':
        return _create_complex_pattern_in_noise(**kwargs)
