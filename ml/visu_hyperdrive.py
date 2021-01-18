#!/usr/bin/env python3

import yaml
import argparse
from pathlib import Path
import pandas as pd

import numpy as np
import dash
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output


def data_bars(df, column):
    COLOR = 'rgba(127, 127, 127, 0.2)'
    if isinstance(df[column][0], (np.bool_,)):
        # return [
        #     {
        #         'if': {
        #             'filter_query': '{{{column}}} == "True"'.format(column=str(column)),
        #             'column_id': column
        #         },
        #         'background': {
        #             f'{COLOR}'
        #         }
        #     }
        # ]
        return []
    elif isinstance(df[column][0], (np.float64, np.int64)):
        n_bins = 100
        bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
        ranges = [
            ((df[column].max() - df[column].min()) * i) + df[column].min()
            for i in bounds
        ]
        styles = []
        for i in range(1, len(bounds)):
            min_bound = ranges[i - 1]
            max_bound = ranges[i]
            max_bound_percentage = bounds[i] * 100
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'background': (
                    f"""
                        linear-gradient(90deg,
                        {COLOR} 0%,
                        {COLOR} {max_bound_percentage}%,
                        rgba(127,127,127,0.0) {max_bound_percentage}%,
                        rgba(127,127,127,0.0) 100%)
                    """
                ),
                'paddingBottom': 2,
                'paddingTop': 2
            })

        return styles
    else:
        return []


def best_error(record):
    return min(record['error'].values())


link_style = {'width': '100%',
              'margin': '30px auto',
              'text-align': 'center',
              'display': 'block'}


def evolutions(records):

    fig = go.Figure(data=[go.Scatter(x=list(r['error'].keys()),
                                     y=list(r['error'].values()),
                                     name=f'run {i}') for i, r in enumerate(records)])
    return dcc.Graph(id='evolutions', figure=fig)


def table_callbacks(app, records):
    @app.callback(Output('evolutions', 'figure'), [Input('table', 'selected_rows')])
    def select(ids):
        if ids is None or len(ids) == 0:
            data = [go.Scatter(x=list(r['error'].keys()),
                               y=list(r['error'].values()),
                               name=f'run {i}') for i, r in enumerate(records)]
        else:
            ids = set(ids)
            data = [go.Scatter(x=list(r['error'].keys()),
                               y=list(r['error'].values()),
                               name=f'run {i}') for i, r in enumerate(records) if i in ids]
        return go.Figure(data=data)


def table_view(records):
    df_hparams = pd.DataFrame(
        [r['hparams'].values() for r in records],
        columns=records[0]['hparams'].keys()
    )

    df_hparams['error'] = pd.Series([best_error(r) for r in records], index=df_hparams.index)
    data_column_names = df_hparams.columns.values
    df_hparams.insert(loc=0, column='id', value=[i for i, _ in enumerate(records)])

    table = dash_table.DataTable(
        id='table',
        columns=[{'name': i, 'id': i, 'format': Format(precision=2, scheme=Scheme.unicode), 'type': 'numeric'}
                 for i in df_hparams.columns],
        data=df_hparams.to_dict('records'),
        sort_action='native',
        row_selectable='multi',
        style_data_conditional=(
            [d for n in data_column_names for d in data_bars(df_hparams, n)]
        ),
    )
    return html.Div([
        evolutions(records),
        table,
        dcc.Link('Paarcoords', href='/paarcoords', style=link_style),
    ], style={'position': 'relative'})


def parcoords_view(records):
    lower = min(best_error(r) for r in records)
    higher = max(best_error(r) for r in records)

    dimension_names = records[0]['hparams'].keys()

    def norm_bool(s):
        return s
        if isinstance(s, bool):
            return str(s)
        return s

    def hparam_range(name):
        series = [norm_bool(r['hparams'][name]) for r in records]
        if isinstance(series[0], str):
            return {'tickvals': list(set(series))}
        else:
            return {'range': [min(series), max(series)]}

    dimensions = [
        {
            'values': [norm_bool(r['hparams'][n]) for r in records],
            'label': n.replace('-', ' ').replace('_', ' '),
            **hparam_range(n)
        }
        for n in dimension_names]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=[best_error(r) for r in records],
                      colorscale='inferno',
                      showscale=True,
                      cmin=-lower,
                      cmax=-higher),
            dimensions=dimensions
        ),
        layout=go.Layout(
            height=700
        )
    )

    return html.Div([
        dcc.Graph(figure=fig),
        dcc.Link('Table', href='/table', style=link_style)
    ])


def main(args):
    with args.filename.open() as f:
        records = yaml.safe_load(f)

    fig = parcoords_view(records)

    table = table_view(records)

    app = dash.Dash()

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='container', children=[table])
    ])

    table_callbacks(app, records)

    @app.callback(Output('container', 'children'), [Input('url', 'pathname')])
    def navigate(pathname):
        if pathname == '/paarcoords':
            return [fig]
        else:
            return [table]

    app.run_server(debug=True, use_reloader=False, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=Path)
    parser.add_argument('--port', default=8050, type=int)
    main(parser.parse_args())
