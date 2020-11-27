#!/usr/bin/env python3

from jinja2 import Template
from pathlib import Path
import random

random.seed(0)

me = Path(__file__)

upper_limit = 20

problems = {}
addition_i = 1
for a in range(1, upper_limit):
    for b in range(1, upper_limit-a):
        problems[f'addition {addition_i}'] = f'{a} + {b} = {a+b}'
        addition_i += 1

for a in range(1, upper_limit):
    for b in range(1, upper_limit-a):
        for c in range(1, upper_limit-a-b):
            problems[f'addition {addition_i}'] = f'{a} + {b} + {c} = {a+b+c}'
            addition_i += 1

multiplication_i = 1
for a in range(1, upper_limit):
    for b in range(1, upper_limit):
        if a*b <= upper_limit:
            problems[f'multiplication {multiplication_i}'] = f'{a} * {b} = {a*b}'
            multiplication_i += 1

for a in range(1, upper_limit):
    for b in range(1, upper_limit-a):
        for c in range(1, upper_limit-a-b):
            if a*b*c <= upper_limit:
                problems[f'multiplication {multiplication_i}'] = f'{a} * {b} * {c} = {a*b*c}'
                multiplication_i += 1

# Take 10% as validation set
validation_keys = set(random.sample(problems.keys(), k=len(problems) // 10))
training_keys = set(problems.keys()) - validation_keys

validation_problems = {k: v for k, v in problems.items() if k in validation_keys}
training_problems = {k: v for k, v in problems.items() if k in training_keys}

with (me.parent / 'dataset.yaml.j2').open() as file:
    Template(file.read()).stream(upper_limit=upper_limit,
                                 name='number-crunching',
                                 training_problems=training_problems,
                                 validation_problems=validation_problems
                                 ).dump(str(me.parent / 'dataset.yaml'))
