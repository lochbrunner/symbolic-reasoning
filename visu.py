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


depth = 2
spread = 2

samples, idents, tags = create_samples_permutation(
    depth=depth, spread=spread)


def load_model(path: str):
    model = TrivialTreeTagger(len(idents), len(tags),
                              embedding_size=32, hidden_size=len(tags))
    print(f'Loading model from {path} ...')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


model = load_model('models/deep.tar')

app = dash.Dash(__name__)


app.layout = html.Div([
    tree_dash_component.TreeDashComponent(
        id='symbol',
        symbol=samples[0][1].as_dict()
    ),
    dcc.Slider(id='selector', min=0, max=len(samples)-1, value=3, step=1),
    html.Button('Prev', id='prev'),
    html.Button('Next', id='next'),
    html.Div([
        html.P(samples[0][0], id='tag'),
        html.P('-', id='tag_prediction'),
        dcc.Graph(id='prediction-heat')
    ])
])


@app.callback(
    Output('selector', 'value'),
    [Input('next', 'n_clicks_timestamp'), Input('prev', 'n_clicks_timestamp')],
    [dash.dependencies.State('selector', 'value')])
def next(next_clicked, prev_clicked, value):
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
    [Input(component_id='selector', component_property='value')]
)
def update_selection(input_value):
    tag, sample = samples[input_value]
    tag_scores = model(sample)
    _, max_index = tag_scores.max(0)
    prediction = f'Prediction {max_index}'

    trace = go.Heatmap(z=[tag_scores.tolist()], colorscale='Electric', colorbar={
                       "title": "Activation"}, showscale=True)

    return sample.as_dict(), f'Tag: {tag}', prediction, {'data': [trace]}


if __name__ == '__main__':
    app.run_server(debug=True)
