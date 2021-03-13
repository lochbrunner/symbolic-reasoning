#!/usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
from common.config_and_arg_parser import ArgumentParser
from common.utils import (get_rule_mapping, get_rule_mapping_by_config,
                          setup_logging, split_dataset)
from common.validation import Error, Ratio
from dataset.bag import BagDataset
from flask import Flask, jsonify, request, send_from_directory
from solver.inferencer import Inferencer
from tqdm import tqdm

from pycore import Scenario, fit


def validate(truth, p, predict, no_negative=False) -> Error:
    error = Error(exact=Ratio(20), exact_no_padding=Ratio(20))
    if no_negative:
        truth = truth*(p+1)/2

    predicted_padding = np.copy(predict)
    predicted_padding[0, :] = np.finfo('f').min

    error.with_padding.update(None, predict, truth)
    error.when_rule.update((truth > 0), predict, truth)
    error.exact.update_global(None, predict, truth)
    error.exact_no_padding.update_global(None, predicted_padding, truth)
    return error


def create_index(inferencer: Inferencer, dataset):
    evaluation_results = []

    def findFirst(array: List[int]) -> int:
        indices = [i for i, e in enumerate(array) if e > 0]
        if len(indices) > 0:
            return indices[0]
        else:
            return len(array)

    for i, (_, _, y, p, v) in tqdm(enumerate(dataset), total=len(dataset.container), desc='indexing', leave=False):
        raw_sample = dataset.container[i]
        initial = raw_sample.initial
        py, _ = inferencer.inference(initial, keep_padding=True)
        validation = validate(truth=y, predict=py, p=p)

        gt_policy = [policy for ruleId, policy in zip(y.tolist(), p.tolist()) if ruleId != 0]
        gt_policy_positive = sum(1 for p in gt_policy if p > 0)
        gt_policy_negative = sum(1 for p in gt_policy if p < 0)

        evaluation_results.append(
            {
                'validation': validation.as_dict(),
                'initial': initial.latex_verbose,
                'contributed': raw_sample.useful,
                'policy_gt': {
                    'positive': gt_policy_positive,
                    'negative': gt_policy_negative
                },
                'index': i,
                'summary': {
                    'exact': findFirst(validation.exact.tops),
                    'exact-no-padding': findFirst(validation.exact_no_padding.tops),
                    'when-rule': findFirst(validation.when_rule.tops),
                    'with-padding': findFirst(validation.with_padding.tops)
                }
            }
        )

    def sort(name):
        extracted = [(i, sample['summary'][name]) for i, sample in enumerate(evaluation_results)]
        extracted = sorted(extracted, key=lambda t: t[1])
        return [i for i, _ in extracted]

    def hist(name):
        hist, bin_edges = np.histogram([(sample['summary'][name])
                                        for sample in evaluation_results], bins=list(range(21)))
        return {'hist': hist.tolist(), 'bin_edges': bin_edges.tolist()}

    indices = {
        'exact': sort('exact'),
        'exact-no-padding': sort('exact-no-padding'),
        'when-rule': sort('when-rule'),
        'with-padding': sort('with-padding'),
    }

    histogram = {
        'exact': hist('exact'),
        'exact_no_padding': hist('exact-no-padding'),
        'when_rule': hist('when-rule'),
        'with_padding': hist('with-padding'),
    }

    return evaluation_results, indices, histogram


def main(config, options):
    setup_logging(**vars(options))

    root_path = Path(__file__).absolute().parents[2]
    dist_folder = root_path / 'dist/client'

    app = Flask(__name__, static_url_path=str(dist_folder))

    dataset = BagDataset.load(config.files.solver_trainings_data)
    scenario = Scenario.load(config.files.scenario)
    rule_mapping = get_rule_mapping(scenario)
    inferencer = Inferencer(config, scenario, fresh_model=False)
    embed2ident = dataset.embed2ident
    assert dataset.ident_dict == inferencer.ident_dict, 'Inconsistent idents in model and dataset'

    overview_samples, indices, histogram_data = create_index(inferencer, dataset)

    @app.route('/')
    def root():
        return send_from_directory(dist_folder, 'index.html')

    @app.route('/<path:path>')
    def assets(path):
        print(f'path: {path}')
        return send_from_directory(dist_folder, path)

    @app.route('/api/sample/<path:index>')
    def sample(index):
        index = int(index)
        raw_sample = dataset.container[index]
        initial = raw_sample.initial
        x, s, y, p, v = dataset[index]
        py, pv = inferencer.inference(initial, keep_padding=True)
        parts_path = [p for p, _ in initial.parts_bfs_with_path]
        highlight_color = '#000077'
        off_focus = ('#aaaaaa', [])

        # Investigate all possible fits
        def hashPath(path: list):
            return '/'.join(str(p) for p in path)
        path2id = {hashPath(path): i for i, path in enumerate(parts_path)}
        possibleFits = []
        for ruleId, rule in rule_mapping.items():
            for fitmap in fit(initial, rule.condition):
                locId = path2id[hashPath(fitmap.path)]
                possibleFits.append({'ruleId': ruleId, 'path': locId})

        return jsonify({
            'latex': initial.latex_verbose,
            'index': index,
            'idents': [embed2ident[embed] for embed in x[:, 0].tolist()],
            'isOperator': [iv == 1 for iv in x[:, 1].tolist()],
            'isFixed': [iv == 1 for iv in x[:, 2].tolist()],
            'isNumber': [iv == 1 for iv in x[:, 3].tolist()],
            'indexMap': s.tolist(),
            'policy': [{'ruleId': ruleId, 'policy': policy, 'path': i} for i, (ruleId, policy) in enumerate(zip(y.tolist(), p.tolist())) if ruleId != 0],
            'groundTruthValue': v.tolist()[0],
            'predictedValue': pv.tolist(),
            'predictedPolicy': py.tolist(),
            'parts': [initial.latex_with_colors([(highlight_color, [])])] +
            [initial.latex_with_colors([off_focus, (highlight_color, path)]) for path in parts_path[1:]],
            'rules': [rule.latex_verbose for rule in dataset.get_rules_raw()],
            'possibleFits': possibleFits,
            'validationMetrics': validate(truth=y, predict=py, p=p).as_dict()
        })

    @app.route('/api/length')
    def sample_count():
        return jsonify({'length': len(dataset.container)})

    @app.route('/api/overview')
    def overview():
        begin = int(request.args['begin'])
        end = int(request.args['end'])
        sorting_key = request.args.get('key', 'none').lower()
        sorting_up = request.args.get('up', 'true')
        dt = 1
        if sorting_up == 'true':
            begin = -begin - 1
            end = -end - 1
            dt = -1
        if sorting_key == 'none':
            request_overview = overview_samples[begin:end:dt]
        else:
            if sorting_key not in indices:
                raise NotImplementedError(f'Not supported sorting: "{sorting_key}"')
            index_map = indices[sorting_key]
            request_overview = [overview_samples[i] for i in index_map[begin:end:dt]]
        return jsonify(request_overview)

    @app.route('/api/histogram')
    def histogram():
        return jsonify(histogram_data)

    app.run()


if __name__ == "__main__":
    parser = ArgumentParser('-c', '--config-file', exclude="scenario-*",
                            prog='deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    main(*parser.parse_args())
