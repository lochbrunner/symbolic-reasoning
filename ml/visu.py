#!/usr/bin/env python3

# system
from typing import List, Set, Dict, Tuple, Optional  # ignore unused-import
import argparse
import yaml
import logging

# dash
# import tree_dash_component
import dash
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
from dash_katex import DashKatex
from activation_dash_component import ActivationDashComponent

# numpy
import numpy as np

# torch
import torch

# local
from common.node import Node
from common.utils import Compose
from common import io
# from dataset.transformers import Embedder, ident_to_id
# from dataset.generate import SymbolBuilder
from pycore import Symbol, Context, Scenario


# def traverse_for_scores(model, node: Node, ident_to_id: Dict[str, int], activation_name: str = 'scores'):
#     scenario = Scenario.load(config.files.scenario)
#     idents = scenario.idents()
#     ident_to_id = {ident: (value+1) for (value, ident) in enumerate(idents)}
#     builder = SymbolBuilder(node)

#     all_scores = []
#     all_paths = []

#     for path, node in builder.traverse_bfs_path():
#         activations = model.introspect(node, ident_to_id)
#         scores = activations[activation_name].detach().numpy()
#         all_scores.insert(0, scores)
#         path = '/'.join([str(d) for d in path])
#         all_paths.insert(0, f'{node.ident} @ {path}')

#     return all_scores, all_paths


def create_ground_truth_string(sample):
    fits = [f'{fit.rule} @ {fit.path}' for fit in sample.fits]
    return ', '.join(fits)


def create_ground_truth_rule_indices(sample):
    parts_path = [p[0] for p in sample.initial.parts_bfs_with_path]

    def get_loc(fit):
        return [fit.rule, parts_path.index(fit.path)]

    return [get_loc(fit) for fit in sample.fits]


@torch.no_grad()
def predict_path_and_label(model, x, s, *args, **kwargs):

    p = torch.ones(x.shape[:-1])
    y, v = model(x, s, p)

    y = y.squeeze()
    y = y.cpu().numpy()
    scores = y
    prediction = np.argmax(x, axis=0)
    x = np.transpose(x)

    return prediction, x, scores


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    files = config['files']

    dataset, model, _, scenario_params = io.load(files['model'])
    model.eval()

    app = dash.Dash(__name__)
    app.title = 'Tree Segmenter Visualization'

    app.layout = html.Div([
        html.H2(id='title', children=model.__class__.__name__),
        html.Div(style={}, className='tree-container', children=[
            # tree_dash_component.TreeDashComponent(
            #     id='symbol',
            #     symbol=dataset.get_node(0).as_dict()
            # ),
            # tree_dash_component.TreeDashComponent(
            #     id='pattern-1',
            #     symbol=Node('?').as_dict()
            # ),
            DashKatex(expression=dataset.container[0].initial.latex, id='initial'),
            html.Div(id='gt-container', children=[
                html.Div(dataset.get_rule_of_sample(0).name, id='gt-rule-name'),
                DashKatex(expression=dataset.get_rule_of_sample(0).latex, id='gt-rule')
            ]),
            html.Div(id='rule-container', children=[
                html.Div(dataset.get_rule_raw(0).name, id='rule-name'),
                DashKatex(expression=dataset.get_rule_raw(0).latex, id='pattern')
            ]),

        ]),
        dcc.Slider(id='rule-selector', min=0, max=dataset.tag_size, value=3, step=1),
        dcc.Slider(id='selector', min=0, max=len(dataset)-1, value=3, step=1),
        html.Button('Prev', id='prev', style={'marginRight': '10px'}),
        html.Button('Next', id='next'),
        html.Span(f'rule: {dataset[0][1][1]}', id='tag', style={'paddingLeft': 10}),
        dcc.Input(id='input-initial', type='text', placeholder='Initial symbol'),
        html.Button('Test', id='test'),
        html.Div([
            html.P('-', id='tag_prediction'),
            dcc.Dropdown(id='activation-selector', options=[
                {'label': name, 'value': name} for name in model.activation_names()],
                value='scores'),
            ActivationDashComponent(id='prediction-heat')
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

    global last_test_button_time
    last_test_button_time = None

    @app.callback(
        [
            # Output(component_id='symbol', component_property='symbol'),
            # Output(component_id='pattern-1', component_property='symbol'),
            Output(component_id='tag', component_property='children'),
            Output(component_id='tag_prediction', component_property='children'),
            Output(component_id='prediction-heat', component_property='data'),
            Output(component_id='initial', component_property='expression'),
            Output(component_id='pattern', component_property='expression'),
            Output(component_id='rule-name', component_property='children'),
            Output(component_id='gt-rule', component_property='expression'),
            Output(component_id='gt-rule-name', component_property='children')


        ],
        [
            Input(component_id='selector', component_property='value'),
            Input(component_id='rule-selector', component_property='value'),
            Input(component_id='activation-selector', component_property='value'),
            Input('test', 'n_clicks_timestamp')
        ],
        [State('input-initial', 'value')]
    )
    def update_selection(sample_id, rule_id, activation_name, test_user_input, user_initial):
        global last_test_button_time
        x_label = [f'{rule.latex}' for rule in dataset.get_rules_raw()]
        x_label[0] = '\\text{padding}'

        def create_color(value):
            v = int(value)
            return "#{0:0{1}x}0000".format(v, 2)

        if test_user_input is None or test_user_input == last_test_button_time:
            initial = dataset.container[sample_id].initial
            sample = dataset.get_sample(sample_id)
            node = dataset.get_node(sample_id)
            ground_truth = 'Ground truth: ' + create_ground_truth_string(dataset.container[sample_id])
            prediction, tag_scores, scores = predict_path_and_label(model, *sample)
            prediction = f'Predict arg-max per sub-tree: {prediction}'
            pattern = Node('?')
            y_label = [f'{part.latex}' for part in initial.parts_bfs]
            y_label.append('\\text{padding}')

            parts_path = [p[0] for p in initial.parts_bfs_with_path]

            rule_scores = scores[rule_id, :]
            rule_scores = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())*255

            colors = [(create_color(score), path)
                      for path, score in zip(parts_path, rule_scores)]

            colored_initial = initial.latex_with_colors(colors)
            rule = dataset.get_rule_raw(rule_id)

            gt_rule = dataset.get_rule_of_sample(sample_id)

            rules_coords = create_ground_truth_rule_indices(dataset.container[sample_id])
            prediction_heat = {'xlabel': x_label, 'ylabel': y_label, 'values': tag_scores, 'markings': rules_coords}

            # return node.as_dict(), pattern.as_dict(), ground_truth, prediction, prediction_heat, colored_initial, rule.latex, rule.name, gt_rule.latex, gt_rule.name
            return ground_truth, prediction, prediction_heat, colored_initial, rule.latex, rule.name, gt_rule.latex, gt_rule.name
        else:
            last_test_button_time = test_user_input
            context = Context.standard()
            initial = Symbol.parse(context, user_initial)
            pattern = Node('?')
            node = Node.from_rust(initial)
            ground_truth = 'Not available'
            x, s, _ = dataset.embed_custom(initial)
            x = torch.unsqueeze(torch.as_tensor(x), 0)
            s = torch.unsqueeze(torch.as_tensor(s), 0)
            prediction, tag_scores, scores = predict_path_and_label(model, x, s)
            y_label = [f'{part.latex}' for part in initial.parts_bfs]
            prediction_heat = {'xlabel': x_label, 'ylabel': y_label, 'values': tag_scores, 'markings': []}

            # Colored initial
            rule_scores = scores[rule_id, :]
            rule_scores = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())*255
            parts_path = [p[0] for p in initial.parts_bfs_with_path]

            colors = [(create_color(score), path)
                      for path, score in zip(parts_path, rule_scores)]

            colored_initial = initial.latex_with_colors(colors)
            rule = dataset.get_rule_raw(rule_id)

            # return node.as_dict(), pattern.as_dict(), ground_truth, prediction, prediction_heat, colored_initial, rule.latex, rule.name, '\\text{n.a.}', 'n.a.'
            return ground_truth, prediction, prediction_heat, colored_initial, rule.latex, rule.name, '\\text{n.a.}', 'n.a.'

    app.run_server(debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualalisation')
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    main(args)
