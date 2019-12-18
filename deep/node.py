from io import StringIO
import json


class NodeEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        return o.__dict__


class Node:
    def __init__(self, ident=None, childs=None, label=None):
        self.ident = ident
        self.childs = childs or []
        self.label = label

    def __repr__(self):
        return json.dumps(self, cls=NodeEncoder)

    def _str_ident(self, buffer, indent):
        ident = self.ident or '?'
        s = ' '
        indents = s * indent
        label = f'{s*self.depth}{self.label}' if self.label is not None else ''
        buffer.write(f'{indents}{ident}{label}\n')
        for child in self.childs:
            child._str_ident(buffer, indent + 1)

    def __str__(self):
        buffer = StringIO()
        self._str_ident(buffer, 0)

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
