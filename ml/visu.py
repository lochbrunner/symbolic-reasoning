#!/usr/bin/env python3

# system
from typing import List, Set, Dict, Tuple, Optional  # ignore unused-import
import logging

# dash
import tree_dash_component
import dash
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from dash_katex import DashKatex

# numpy
import numpy as np

# torch
import torch

# local
from common.node import Node
from common.utils import Compose
from common import io
from dataset.transformers import Embedder, ident_to_id
from dataset.generate import SymbolBuilder


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def traverse_for_scores(model, node: Node, activation_name: str = 'scores'):
    builder = SymbolBuilder(node)

    all_scores = []
    all_paths = []

    for path, node in builder.traverse_bfs_path():
        activations = model.introspect(node, ident_to_id)
        scores = activations[activation_name].detach().numpy()
        all_scores.insert(0, scores)
        path = '/'.join([str(d) for d in path])
        all_paths.insert(0, f'{node.ident} @ {path}')

    return all_scores, all_paths


dataset, model, _, scenario_params = io.load('../snapshots/cnn_tree_bag.sp', transform=Compose([]))
model.eval()


def create_node(idents):
    depth = scenario_params.pattern_depth
    spread = scenario_params.spread
    builder = SymbolBuilder()
    for _ in range(depth):
        builder.add_level_uniform(spread)
    builder.set_idents_bfs(idents)
    return builder.symbol


def ground_truth_path(node):
    builder = SymbolBuilder(node)

    for path, node in builder.traverse_bfs_path():
        if node.label and node.label > 0:
            return path, node.label
    return None, None


@torch.no_grad()
def predict_path_and_label(model, x, y, m):

    x = model(x)

    x = x.squeeze()
    x = x.cpu().numpy()
    scores = x
    mask = Embedder.leaf_mask(scenario_params)
    mask_indices = np.squeeze(np.argwhere(mask == 1))
    x = x[:, mask_indices]
    prediction = np.argmax(x, axis=0)
    x = np.transpose(x)

    # Find strongest non 0 activation
    paths = np.array(list(Embedder.legend(scenario_params)))

    paths = paths[mask_indices]

    return prediction, paths, x, scores


def predict(model, node):
    scores = model.introspect(node, ident_to_id)['scores']
    _, i = scores.max(0)
    return i.item()


app = dash.Dash(__name__)
app.title = 'Tree Segmenter Visualization'


app.layout = html.Div([
    html.H2(id='title', children=model.__class__.__name__),
    html.Div(style={}, className='tree-container', children=[
        tree_dash_component.TreeDashComponent(
            id='symbol',
            symbol=dataset.get_node(0).as_dict()
        ),
        tree_dash_component.TreeDashComponent(
            id='pattern-1',
            symbol=Node('?').as_dict()
        ),
        DashKatex(expression=dataset.raw_samples[0][0].latex, id='initial'),
        html.Div(id='rule-container', children=[
            html.Div(dataset.get_rule_raw(0).name, id='rule-name'),
            DashKatex(expression=dataset.get_rule_raw(0).latex, id='pattern')
        ]),
        html.Div(id='gt-container', children=[
            html.Div(dataset.get_rule_of_sample(0).name, id='gt-rule-name'),
            DashKatex(expression=dataset.get_rule_of_sample(0).latex, id='gt-rule')
        ])

    ]),
    dcc.Slider(id='rule-selector', min=0, max=dataset.tag_size, value=3, step=1),
    dcc.Slider(id='selector', min=0, max=len(dataset)-1, value=3, step=1),
    html.Button('Prev', id='prev', style={'marginRight': '10px'}),
    html.Button('Next', id='next'),
    html.Span(f'rule: {dataset[0][1][1]}', id='tag', style={'paddingLeft': 10}),
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
        return min(value + 1, len(dataset) - 1)
    else:
        return max(0, value - 1)


@app.callback(
    [
        Output(component_id='symbol', component_property='symbol'),
        Output(component_id='pattern-1', component_property='symbol'),
        Output(component_id='tag', component_property='children'),
        Output(component_id='tag_prediction', component_property='children'),
        Output(component_id='prediction-heat', component_property='figure'),
        Output(component_id='initial', component_property='expression'),
        Output(component_id='pattern', component_property='expression'),
        Output(component_id='rule-name', component_property='children'),
        Output(component_id='gt-rule', component_property='expression'),
        Output(component_id='gt-rule-name', component_property='children')


    ],
    [
        Input(component_id='selector', component_property='value'),
        Input(component_id='rule-selector', component_property='value'),
        Input(component_id='activation-selector', component_property='value')
    ]
)
def update_selection(sample_id, rule_id, activation_name):
    # Tag
    use_tag = False
    if use_tag:
        sample, tag, _ = dataset[sample_id]
        tag_scores, paths = traverse_for_scores(model, sample, activation_name)
        max_index = predict(model, sample)
        prediction = f'Prediction {max_index}'
        ground = f'Tag: {tag}'
        pattern = Node('-')
        x = None
    else:
        initial = dataset.raw_samples[sample_id][0]
        x, y, m = dataset[sample_id]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        m = m.unsqueeze(0)
        sample = dataset.get_node(sample_id)
        gp, gl = ground_truth_path(sample)
        ground = f'Ground truth: {gl} @ {gp}'
        prediction, paths, tag_scores, scores = predict_path_and_label(model, x, y, m)
        prediction = f'Predict arg-max: {prediction}'
        pattern = Node('?')
        x = [f'$${rule.latex}$$' for rule in dataset.get_rules_raw()]

        # def stringify(path):
        #     return '@' + '/'.join(str(i) for i in path)
        green = '#22ff22'
        # paths = [f'@ $${initial.latex_with_colors([(green, path)])}$$' for path in paths]
        paths = [f'@ $${initial.at(path).latex}$$' for path in paths]
        print(paths)

        legend = Embedder.legend(dataset)
        rule_scores = scores[rule_id, :]
        rule_scores = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())*255

        def create_color(value):
            v = int(value)
            return "#{0:0{1}x}0000".format(v, 2)

        colors = [(create_color(score), path)
                  for path, score in zip(legend, rule_scores)]

        colored_initial = initial.latex_with_colors(colors)
        rule = dataset.get_rule_raw(rule_id)

        gt_rule = dataset.get_rule_of_sample(sample_id)

    trace = go.Heatmap(y=paths, z=tag_scores, x=x, colorscale='Electric', colorbar={
        'title': 'Score'}, showscale=True)

    return sample.as_dict(), pattern.as_dict(), ground, prediction, {'data': [trace]}, colored_initial, rule.latex, rule.name, gt_rule.latex, gt_rule.name


if __name__ == '__main__':
    app.run_server(debug=True)
