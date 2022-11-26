#!/usr/bin/env python3

from pycore import Context, Symbol, Decoration, FitInfo

from typing import Iterator
import unittest
import numpy as np
import numpy.testing as npt


class TestSymbol(unittest.TestCase):
    @staticmethod
    def unroll(x: Symbol) -> Iterator[Symbol]:
        stack = [(x, '/')]
        # Using the path as unique id because id(n) is not unique enough ;)
        # id(n) is the memory address which get reused by the rust backend.
        seen = set()
        while len(stack) > 0:
            n, p = stack[-1]
            if len(n.childs) > 0 and p not in seen:
                stack.extend([(c, f'{p}{i}') for i, c in enumerate(n.childs[::-1])])
                seen.add(p)
            else:
                stack.pop()
                yield n

    def test_dumping(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a/b')

        self.assertEqual(symbol.latex, '\\frac{a}{b}')
        self.assertEqual(str(symbol), 'a/b')
        self.assertEqual(symbol.ident, '/')
        self.assertEqual(str(symbol.at([1])), 'b')

    def test_dumping_with_decoration(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b')
        latex = symbol.latex_with_deco(
            [
                Decoration([], '<C>', '</C>'),
                Decoration([0], '<A>', '</A>'),
                Decoration([1], '<B>', '</B>'),
            ]
        )
        self.assertEqual(latex, '<C><A>a</A>+<B>b</B></C>')

    def test_dumping_with_colors(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b')
        latex = symbol.latex_with_colors([('red', [0])])
        self.assertEqual(latex, '\\textcolor{red}{a}+b')

    def test_dump_latex_verbose(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+(b+c)')
        latex = symbol.latex_verbose.strip()
        self.assertEqual(latex, 'a+\\left( b+c\\right)')

    def test_unroll(self):
        context = Context.standard()

        a = Symbol.parse(context, 'a+b')
        self.assertListEqual([str(p) for p in TestSymbol.unroll(a)], ['a', 'b', 'a+b'])

        a = Symbol.parse(context, 'a')
        a.pad('<PAD>', 2, 1)
        self.assertListEqual([str(p) for p in a.childs], ['<PAD>', '<PAD>'])

    def test_padding(self):

        context = Context.standard()

        a = Symbol.parse(context, 'x=0*x/1')
        a = a.create_padded('<PAD>', 2, 5)
        self.assertEqual(63, a.tree.count('\n'))
        unrolled = TestSymbol.unroll(a)
        exprected_len = sum(2**l for l in range(6))
        self.assertEqual(len(list(unrolled)), exprected_len)

    def test_comparison(self):

        context = Context.standard()

        a = Symbol.parse(context, 'a+c')
        b = Symbol.parse(context, 'a+b')
        a1 = Symbol.parse(context, 'a+c')
        self.assertNotEqual(a, b)
        self.assertEqual(a, a1)

        self.assertNotEqual(hash(a), hash(b))
        self.assertNotEqual(id(a), id(b))

        self.assertEqual(hash(a), hash(a1))
        self.assertNotEqual(id(a), id(a1))

    def test_attributes(self):
        context = Context.standard()
        a = Symbol.parse(context, 'a+c')

        a.label = 'my_label'
        self.assertEqual(a.label, 'my_label')

        del a.label
        with self.assertRaises(KeyError):
            a.label

    def test_traverse_dfs(self):
        context = Context.standard()

        a = Symbol.parse(context, 'a*b+c*d')

        self.assertListEqual(
            [str(p) for p in a.parts_dfs], ['a*b+c*d', 'c*d', 'd', 'c', 'a*b', 'b', 'a']
        )

    def test_traverse_bfs(self):
        context = Context.standard()
        a = Symbol.parse(context, 'a*b+c*d')

        self.assertListEqual(
            [str(p) for p in a.parts_bfs], ['a*b+c*d', 'a*b', 'c*d', 'a', 'b', 'c', 'd']
        )

    def test_traverse_bfs_with_path(self):
        context = Context.standard()
        a = Symbol.parse(context, 'a*b+c*d')
        paths = [p[0] for p in a.parts_bfs_with_path]
        self.assertListEqual(paths, [[], [0], [1], [0, 0], [0, 1], [1, 0], [1, 1]])

    def test_unrolled_cnn_embed(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b=c*d')
        embed_dict = {'=': 1, '+': 2, '*': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7}
        fits = [FitInfo(2, [0], True)]
        spread = 2
        useful = True
        index_encoding = True
        positional_encoding = False
        target_size = 3
        (
            embedding,
            indices,
            positional_encoding,
            label,
            policy,
            value,
            target,
            mask,
        ) = symbol.embed(
            embed_dict,
            0,
            spread,
            symbol.depth,
            target_size,
            fits,
            useful,
            index_encoding,
            positional_encoding,
            use_additional_features=True,
        )

        self.assertIsNone(positional_encoding)
        npt.assert_equal(embedding[:, 0], [1, 2, 3, 4, 5, 6, 7, 0])
        npt.assert_equal(embedding[:, 1], [1, 1, 1, 0, 0, 0, 0, 0])  # is operator
        npt.assert_equal(embedding[:, 2], [1, 1, 1, 0, 0, 0, 0, 0])  # is fixed
        npt.assert_equal(embedding[:, 3], [0, 0, 0, 0, 0, 0, 0, 0])  # is number
        npt.assert_equal(
            indices,
            [
                [0, 1, 2, 7],
                [1, 3, 4, 0],
                [2, 5, 6, 0],
                [3, 7, 7, 1],
                [4, 7, 7, 1],
                [5, 7, 7, 2],
                [6, 7, 7, 2],
                [7, 7, 7, 7],
            ],
        )
        npt.assert_equal(label, [0, 2, 0, 0, 0, 0, 0, 0])
        npt.assert_equal(policy, [0, 1.0, 0, 0, 0, 0, 0, 0])
        npt.assert_equal(value, [1])

        self.assertEqual(
            target.tolist(),
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(
            mask.tolist(),
            [
                [False, False, False],
                [False, False, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ],
        )

    def test_graph_embed(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b=c*d')
        embed_dict = {'=': 1, '+': 2, '*': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7}
        fits = [FitInfo(1, [0], True)]

        actual = symbol.create_graph_embedding(
            embed_dict,
            2,
            fits,
            True,
            use_additional_features=True,
        )

        npt.assert_equal(
            actual.nodes,
            np.array(
                [
                    [1, 1, 1, 0],
                    [2, 1, 1, 0],
                    [3, 1, 1, 0],
                    [4, 0, 0, 0],
                    [5, 0, 0, 0],
                    [6, 0, 0, 0],
                    [7, 0, 0, 0],
                ]
            ),
        )
        npt.assert_equal(
            actual.receivers,
            np.array([0, 0, 1, 1, 2, 2, 1, 2, 3, 4, 5, 6]),
        )
        npt.assert_equal(
            actual.senders,
            np.array([1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2]),
        )
        self.assertEqual(actual.n_node, 7)
        self.assertEqual(actual.n_edge, 12)
        npt.assert_equal(actual.value, np.array([1]))
        npt.assert_equal(
            actual.target,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        )
        npt.assert_equal(
            actual.mask,
            np.array(
                [
                    [False, False],
                    [False, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            ),
        )

    def test_size(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b=c*d')

        self.assertEqual(symbol.size, 7)

    def test_number_of_embedded_properties(self):
        self.assertEqual(Symbol.number_of_embedded_properties, 3)

    def test_memory_usage(self):
        context = Context.standard()
        symbol = Symbol.parse(context, 'a+b=c*d')
        self.assertLess(symbol.memory_usage, 1000)

    def test_replace_and_pad(self):
        context = Context.standard()
        context.add_function('sqrt', True)
        context.add_function('root', True)
        symbol = Symbol.parse(context, 'sqrt(a+b)=c')
        actual = symbol.replace_and_pad('sqrt', 'root', 2, Symbol.number(2))
        self.assertEqual(str(actual), 'root(a+b, 2)=c')


if __name__ == '__main__':
    unittest.main()
