from io import StringIO
import json
import unittest


class NodeEncoder(json.JSONEncoder):
    '''Needed for json dumps'''

    def default(self, o):  # pylint: disable=E0202
        return o.__dict__


class Node:
    '''Python implementation of the Symbol'''

    def __init__(self, ident=None, childs=None, label=None, parent=None):
        self.ident = ident
        self.childs = childs or []
        self.label = label
        self.parent = parent

    def __repr__(self):
        return json.dumps(self, cls=NodeEncoder)

    def str_ident(self, buffer, indent):
        ident = self.ident or '?'
        s = ' '
        indents = s * indent
        label = f'{s*self.depth}{self.label}' if self.label is not None else ''
        buffer.write(f'{indents}{ident}{label}\n')
        for child in self.childs:
            child.str_ident(buffer, indent + 1)

    def __str__(self):
        buffer = StringIO()
        self.str_ident(buffer, 0)

        return buffer.getvalue()

    def __eq__(self, other):
        return self.ident == other.ident and self.childs == other.childs

    def __hash__(self):
        return hash(str(self))

    @property
    def depth(self):
        depth = -1
        stack = [self]
        while len(stack) > 0:
            node = stack.pop()
            depth += 1
            if len(node.childs) > 0:
                stack += node.childs[0:1]
            else:
                break

        return depth

    def as_dict(self):
        return {'ident': self.ident, 'childs': [child.as_dict() for child in self.childs]}


class TestSymbol(unittest.TestCase):
    '''Unit tests for python Symbol class'''

    def test_equality_small(self):
        a = Node('a')
        b = Node('a')
        self.assertEqual(a, b)

    def test_inequality_small(self):
        a = Node('a')
        b = Node('b')
        self.assertNotEqual(a, b)

    def test_equality_large(self):
        a = Node('a', childs=[Node('b'), Node('c')])
        b = Node('a', childs=[Node('b'), Node('c')])
        self.assertEqual(a, b)

    def test_inequality_large(self):
        a = Node('a', childs=[Node('b'), Node('c')])
        b = Node('a', childs=[Node('b'), Node('a')])
        self.assertNotEqual(a, b)
