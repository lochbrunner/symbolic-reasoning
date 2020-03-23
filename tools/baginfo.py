#!/usr/bin/env python3

from pycore import Bag

import argparse


class Table:
    class Row:
        def __init__(self, columns):
            self.columns = columns

    class Seperator:
        pass

    def __init__(self):
        self.rows = []

    def add_sep(self):
        self.rows.append(Table.Seperator())

    def add_row(self, name, value):
        self.rows.append(Table.Row([str(name), str(value)]))

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
            html = ''
            html += f'<h2>file: {args.filename}</h2>'
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
                    text-align: left;
                }

                th:nth-child(2),
                td:nth-child(2) {
                    text-align: right;
                }
            </style>'''

            html += '''<tr>
                <th>name</th>
                <th>count</th>
            </tr>'''

            for row in self.rows:
                if type(row) is Table.Row:
                    html += f'''<tr>
                        <td>{row.columns[0]}</td>
                        <td>{row.columns[1]}</td>
                    </tr>'''
                else:
                    html += '''<tr>
                        <td colspan="2"><hr/></td>
                    </tr>'''

            html += '</table>'

            print(html)


def main(args):
    bag = Bag.load(args.filename)

    table = Table()

    rules_tuple = list(zip(bag.meta.rules, bag.meta.rule_distribution))
    rules_tuple.sort(key=lambda r: r[1], reverse=True)

    for (rule, count) in rules_tuple:
        table.add_row(f'{rule.name}:', count)

    table.add_sep()
    total = sum(bag.meta.rule_distribution[1:])
    table.add_row('total:', total)
    table.add_row('idents:', len(bag.meta.idents))

    table.print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('baginfo')
    parser.add_argument('--format', choices=['plain', 'html'], default='plain')
    parser.add_argument('filename')
    args = parser.parse_args()

    main(parser.parse_args())
