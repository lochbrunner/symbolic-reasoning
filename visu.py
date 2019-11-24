#!/usr/bin/env python3

# system
from typing import List, Set, Dict, Tuple, Optional

# dash
import tree_dash_component
import dash
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

# torch
import torch

# local
from deep.node import Node
from deep.model import TreeTagger, TrivialTreeTagger
from deep.generate import create_samples_permutation
from deep.generate import SymbolBuilder


depth = 2
spread = 2

samples, idents, tags = create_samples_permutation(
    depth=depth, spread=spread)


def load_model(path: str):
    checkpoint = torch.load(path)
    device = torch.device('cpu')
    model_hyper_parameter = checkpoint['hyper_parameter']
    model = TrivialTreeTagger(len(idents), len(tags), device,
                              model_hyper_parameter)
    print(f'Loading model from {path} ...')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def traverse_for_scores(model, node: Node, activation_name: str = 'scores'):
    builder = SymbolBuilder(node)

    all_scores = []
    all_paths = []

    for path, node in builder.traverse_bfs_path():
        vars = model.introspect(node)
        scores = vars[activation_name]
        all_scores.insert(0, scores)
        path = '/'.join([str(d) for d in path])
        all_paths.insert(0, f'{node.ident} @ {path}')

    return all_scores, all_paths


model = load_model('models/deep.tar')

app = dash.Dash(__name__)
app.title = 'TreeLstm Visualization'


app.layout = html.Div([
    tree_dash_component.TreeDashComponent(
        id='symbol',
        symbol=samples[0][1].as_dict()
    ),
    dcc.Slider(id='selector', min=0, max=len(samples)-1, value=3, step=1),
    html.Button('Prev', id='prev', style={'marginRight': '10px'}),
    html.Button('Next', id='next'),
    html.Span(samples[0][0], id='tag', style={'paddingLeft': 10}),
    html.Div([
        html.P('-', id='tag_prediction'),
        dcc.Dropdown(id='activation-selector', options=[
                     {'label': name, 'value': name} for name in model.activation_names()],
                     value='scores'),
        dcc.Graph(id='prediction-heat')
    ])
])


@app.callback(
    Output('selector', 'value'),
    [Input('next', 'n_clicks_timestamp'), Input('prev', 'n_clicks_timestamp')],
    [dash.dependencies.State('selector', 'value')])
def pagination(next_clicked, prev_clicked, value):
    next_clicked = 0 if next_clicked is None else next_clicked
    prev_clicked = 0 if prev_clicked is None else prev_clicked
    forward = next_clicked > prev_clicked
    if forward:
        return min(value + 1, len(samples) - 1)
    else:
        return max(0, value - 1)


@app.callback(
    [Output(component_id='symbol', component_property='symbol'),
     Output(component_id='tag', component_property='children'),
     Output(component_id='tag_prediction', component_property='children'),
     Output(component_id='prediction-heat',  component_property='figure')
     ],
    [Input(component_id='selector', component_property='value'),
     Input(component_id='activation-selector', component_property='value')]
)
def update_selection(sample_id, activation_name):
    tag, sample = samples[sample_id]
    tag_scores, paths = traverse_for_scores(model, sample, activation_name)
    _, max_index = model(sample).max(0)
    prediction = f'Prediction {max_index}'

    trace = go.Heatmap(y=paths, z=tag_scores, colorscale='Electric', colorbar={
                       'title': 'Score'}, showscale=True)

    return sample.as_dict(), f'Tag: {tag}', prediction, {'data': [trace]}


if __name__ == '__main__':
    app.run_server(debug=True)
