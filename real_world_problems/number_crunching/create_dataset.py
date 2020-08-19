#!/usr/bin/env python3

from jinja2 import Template

with open('./dataset.yaml.j2') as file:
    Template(file.read()).stream(upper_limit=20, name='number-crunching').dump('./dataset.yaml')

# rendered = template.render()

# with open('./dataset.yaml', 'w') as file:
#     file.write(rendered)
# or Template('Hello {{ name }}!').stream(name='foo').dump('hello.html')
