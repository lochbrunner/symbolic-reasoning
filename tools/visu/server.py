#!/usr/bin/env python3

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from common.config_and_arg_parser import ArgumentParser
from dataset.bag import BagDataset
from common.utils import setup_logging, get_rule_mapping_by_config
# from ml.pycore import Symbol


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
        initial = dataset.container[index].initial
        return jsonify({'latex': initial.latex, 'index': index})

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
