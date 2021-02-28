#!/usr/bin/env python3

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from common.config_and_arg_parser import ArgumentParser
from dataset.bag import BagDataset
from common.utils import setup_logging, get_rule_mapping_by_config
from common.utils import get_rule_mapping
from solver.inferencer import Inferencer
from pycore import Scenario, fit


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
        py, pv = inferencer.inference(initial)
        parts_path = [p for p, _ in initial.parts_bfs_with_path]
        highlight_color = '#000000'
        off_focus = ('#888888', [])

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
            'parts': [initial.latex] + [initial.latex_with_colors([off_focus, (highlight_color, path)]) for path in parts_path[1:]][:14],
            'rules': [rule.latex_verbose for rule in dataset.get_rules_raw()],
            'possibleFits': possibleFits,
            'validationMetrics': None
        })

    @app.route('/api/sample-overview')
    def sample_count():
        return jsonify({'length': len(dataset.container)})

    app.run()


if __name__ == "__main__":
    parser = ArgumentParser('-c', '--config-file', exclude="scenario-*",
                            prog='deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    main(*parser.parse_args())
