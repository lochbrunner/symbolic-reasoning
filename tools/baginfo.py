#!/usr/bin/env python3

from pycore import Bag

import argparse
from os import path


class TableColumn:
    def __init__(self, content, tooltip=None):
        self.content = content
        self.tooltip = tooltip

    @property
    def html(self):
        if self.tooltip is not None:
            attrs = f'title="{self.tooltip}"'
        else:
            attrs = ''
        return f'<td {attrs}>{self.content}</td>'


class Table:
    class Row:
        def __init__(self, columns, ratio=None):
            self.columns = columns
            self.ratio = ratio

    class Seperator:
        pass

    def __init__(self):
        self.rows = []

    def add_sep(self):
        self.rows.append(Table.Seperator())

    def add_row(self, name, *values, ratio=None):
        def embed(column):
            if isinstance(column, (str, int, float)):
                return TableColumn(column)
            if isinstance(column, TableColumn):
                return column

        self.rows.append(Table.Row([embed(name), *[embed(value) for value in values]], ratio=ratio))

    def print(self, args):
        if args.format == 'plain':
            c1 = max([len(row.columns[0]) for row in self.rows if type(row) is Table.Row]) + 1
            c2 = max([len(row.columns[1]) for row in self.rows if type(row) is Table.Row])

            for row in self.rows:
                if type(row) is Table.Row:
                    name = row.columns[0].ljust(c1)
                    value = str(row.columns[1]).rjust(c2)
                    print(f'{name} {value}')
                else:
                    print('-'*(c1+c2+1))

        elif args.format == 'html':
            rel_path = path.relpath(args.filename, path.join(path.dirname(__file__), '..'))
            html = ''
            html += f'<h2>file: {rel_path}</h2>'
            html += '<table style="width:100%">'
            html += '''<style>
                table,
                th,
                td {
                    /* border: 1px solid black; */
                    /* border-collapse: collapse; */
                }

                th,
                td {
                    padding: 5px;
                }

                th:nth-child(1),
                td:nth-child(1) {
                    width: 80%;
                    text-align: left;
                }

                th:nth-child(2),
                td:nth-child(2) {
                    text-align: right;
                }
                th:nth-child(3),
                td:nth-child(3) {
                    text-align: right;
                }
                th:nth-child(4),
                td:nth-child(4) {
                    text-align: right;
                }
                tr:hover{
                    background: rgba(128, 128, 128, 10%);
                }
            </style>'''

            html += '''<tr>
                <th>name</th>
                <th>count</th>
                <th>id</th>
                <th>p/c</th>
            </tr>'''

            max_columns = max(len(row.columns) for row in self.rows if isinstance(row, Table.Row))

            for row in self.rows:
                if type(row) is Table.Row:
                    rest_columns = '\n'.join(c.html for c in row.columns[1:])
                    if row.ratio is None:
                        html += f'''<tr>
                            {row.columns[0].html}
                            {rest_columns}
                        </tr>'''
                    else:
                        p = f'{(row.ratio*100):.2f}'
                        html += f'''<tr>
                            <td style="background: linear-gradient(to right, rgba(128, 128, 128, 0.25) {p}%, rgba(128, 128, 100, 0.0) {p}%)">{row.columns[0].content}</td>
                            {rest_columns}
                        </tr>'''
                else:
                    html += f'''<tr>
                        <td colspan="{max_columns}"><hr/></td>
                    </tr>'''

            html += '</table>'

            print(html)


def main(args):
    bag = Bag.load(args.filename)

    table = Table()

    rules_tuple = list(enumerate(zip(bag.meta.rules, bag.meta.rule_distribution), 0))
    rules_tuple.sort(key=lambda r: r[1][1][1]+r[1][1][0], reverse=True)
    max_count = max(p+n for p, n in bag.meta.rule_distribution[1:])

    for i, (rule, (p_count, n_count)) in rules_tuple[0:]:
        count = p_count + n_count
        rel_pos = f'{p_count/count*100.:.1f}%' if count > 0 else '-'
        rel_pos = TableColumn(rel_pos, tooltip=f'positive: {p_count}\nnegative: {n_count}')
        table.add_row(TableColumn(rule.name), TableColumn(count), TableColumn(f'#{i}'), rel_pos, ratio=count/max_count)

    table.add_sep()
    total = sum(p+n for p, n in bag.meta.rule_distribution[1:])
    table.add_row('total:', total)
    table.add_row('idents:', ', '.join(bag.meta.idents))
    table.add_row('values:', ', '.join([f'{v}' for v in bag.meta.value_distribution]))

    table.print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('baginfo')
    parser.add_argument('--format', choices=['plain', 'html'], default='plain')
    parser.add_argument('filename')
    args = parser.parse_args()

    main(parser.parse_args())
