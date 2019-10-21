from io import StringIO
import json


class NodeEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        return o.__dict__


class Node:
    def __init__(self, ident=None, childs=[]):
        self.ident = ident
        self.childs = childs

    def __repr__(self):
        return json.dumps(self, cls=NodeEncoder)

    def _str_ident(self, buffer, ident):
        buffer.write(' ' * ident + (self.ident or '?') + '\n')
        for child in self.childs:
            child._str_ident(buffer, ident + 1)

    def __str__(self):
        buffer = StringIO()
        self._str_ident(buffer, 0)

        return buffer.getvalue()

    def __eq__(self, other):
        return self.ident == other.ident and self.childs == other.childs

    def __hash__(self):
        return hash(str(self))
