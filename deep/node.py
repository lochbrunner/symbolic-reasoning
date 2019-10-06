from io import StringIO
import json


class NodeEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Node:
    def __init__(self):
        self.childs = []
        self.ident = None

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
