#!/usr/bin/env python3

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from common.config_and_arg_parser import ArgumentParser
from dataset.bag import BagDataset
from common.utils import setup_logging, get_rule_mapping_by_config


def main(config, options):
    setup_logging(**vars(options))

    root_path = Path(__file__).absolute().parents[2]
    dist_folder = root_path / 'dist/client'

    app = Flask(__name__, static_url_path=str(dist_folder))

    dataset = BagDataset.load(config.files.solver_trainings_data)

    @app.route('/')
    def root():
        return send_from_directory(dist_folder, 'index.html')

    @app.route('/<path:path>')
    def send_js(path):
        print(f'path: {path}')
        return send_from_directory(dist_folder, path)

    @app.route('/api/sample/<path:index>')
    def sample(index):
        index = int(index)
        raw_sample = dataset.container[index]
        initial = raw_sample.initial
        x, s, y, p, v = dataset[index]
        parts_path = [p for p, _ in initial.parts_bfs_with_path]
        highlight_color = '#000000'
        off_focus = ('#888888', [])
        return jsonify({
            'latex': initial.latex,
            'index': index,
            'idents': x.tolist(),
            'indexMap': s.tolist(),
            'policy': [{'ruleId': ruleId, 'policy': policy, 'path': i} for i, (ruleId, policy) in enumerate(zip(y.tolist(), p.tolist())) if ruleId != 0],
            'value': v.tolist()[0],
            'predictedValue': None,
            'parts': [initial.latex] + [initial.latex_with_colors([off_focus, (highlight_color, path)]) for path in parts_path[1:]][:14],
            'rules': [rule.latex for rule in dataset.get_rules_raw()],
            'predictions': [[]],
            'possibilities': []
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
