#!/usr/bin/env python3

from pathlib import Path
from typing import List, Dict
import sqlite3

import numpy as np
from common.config_and_arg_parser import ArgumentParser
from common.utils import (get_rule_mapping, get_rule_mapping_by_config,
                          setup_logging, split_dataset)
from training.validation import Error, Ratio
from dataset.bag import BagDataset
from flask import Flask, jsonify, request, send_from_directory, g
from solver.inferencer import Inferencer
from tqdm import tqdm

from pycore import Scenario, fit

database_path = Path('/tmp/database.db')


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(database_path)
    return db


def validate(truth, predict, no_negative=False) -> Error:
    error = Error(exact=Ratio(20), exact_no_padding=Ratio(20))
    # if no_negative:
    #     truth = truth*(p+1)/2
    predicted_padding = np.copy(predict)
    predicted_padding[0, :] = np.finfo('f').min

    error.with_padding.update(None, predict, truth)
    error.when_rule.update((truth > 0), predict, truth)
    error.exact.update_global(None, predict, truth)
    error.exact_no_padding.update_global(None, predicted_padding, truth)
    return error


def query(begin: int, end: int, up: bool, sorting_key: str, rule_filter: str, rule_id2verbose: Dict[int, str]):
    con = get_db()
    cur = con.cursor()
    if up:
        sorting = 'ASC'
    else:
        sorting = 'DESC'

    key_map = {
        'exact-no-padding': 'summary__exact_no_padding',
        'exact': 'summary__exact',
        'when-rule': 'summary__when_rule',
        'with-padding': 'summary__with_padding',
        'value-gt': 'value__gt',
        'value-error': 'value__error',
        'name': 'initial',
        'positive': 'policy_gt__positive',
        'negative': 'policy_gt__negative',
        'na': 'possibilities',
    }

    sorting_key = key_map.get(sorting_key, 'initial')

    if rule_filter != '':
        rule_ids = [str(i) for i, verbose in rule_id2verbose.items() if rule_filter in verbose]

        rule_ids = ', '.join(rule_ids)

        cur.execute(
            f'SELECT * FROM samples inner join rules on rules.sample_id = samples.id where rules.id in ({rule_ids}) order by {sorting_key} {sorting} limit {begin}, {end}')

    else:
        cur.execute(
            f'SELECT * FROM samples order by {sorting_key} {sorting} limit {begin}, {end}')

    return [
        {
            'initial': row[0],
            'value': {
                'gt': row[1] == 1,
                'predicted': row[2],
                'error': row[3]
            },
            'policy_gt': {
                'positive': row[4],
                'negative': row[5]
            },
            'possibilities': row[6],
            'index': row[7],
            'summary': {
                'exact': row[8],
                'exact-no-padding': row[9],
                'when-rule': row[10],
                'with-padding': row[11]
            }
        }
        for row in cur.fetchall()]


def create_index(inferencer: Inferencer, dataset):
    sqlite3.register_adapter(bool, int)
    sqlite3.register_converter('boolean', lambda v: bool(int(v)))
    if database_path.exists():
        database_path.unlink()
    con = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute('CREATE TABLE samples ('
                'initial text, '
                'value__gt boolean, '
                'value__predicted real, '
                'value__error real, '
                'policy_gt__positive integer, '
                'policy_gt__negative integer, '
                'possibilities integer, '
                'id integer primary key, '
                'summary__exact integer, '
                'summary__exact_no_padding integer, '
                'summary__when_rule integer, '
                'summary__with_padding integer '
                ')')
    cur.execute('create table rules (id integer, sample_id integer)')
    con.commit()

    # evaluation_results = []

    def findFirst(array: List[int]) -> int:
        indices = [i for i, e in enumerate(array) if e > 0]
        if len(indices) > 0:
            return indices[0]
        else:
            return len(array)

    evaluation_results = []

    for i, (_, _, y, p, v, target, mask) in tqdm(enumerate(dataset), total=len(dataset.container), desc='indexing', leave=False):
        raw_sample = dataset.container[i]
        initial = raw_sample.initial
        py, pv = inferencer.inference(initial, keep_padding=True)
        py = np.transpose(py, (1, 0))
        validation = validate(truth=y, predict=py)

        gt_policy_positive = np.count_nonzero(target > 0.5)
        gt_policy_negative = np.count_nonzero(target < -0.5)
        possibilities = np.count_nonzero(mask)

        cur.execute(
            "INSERT INTO samples VALUES("
            f"'{initial.latex_verbose}', "                      # initial
            f"{raw_sample.useful}, "                            # value
            f"{pv.tolist()}, "
            f"{abs(pv.tolist() - v[0].tolist())}, "
            f"{gt_policy_positive}, "
            f"{gt_policy_negative}, "
            f"{possibilities}, "
            f"{i}, "                                            # index
            f"{findFirst(validation.exact.tops)}, "             # summary
            f"{findFirst(validation.exact_no_padding.tops)}, "  # summary
            f"{findFirst(validation.when_rule.tops)}, "         # summary
            f"{findFirst(validation.with_padding.tops)} "      # summary
            ")")

        for ruleId in [ruleId for ruleId, policy in zip(y.tolist(), p.tolist()) if ruleId != 0 and policy > 0]:
            cur.execute(f'insert into rules values({ruleId}, {i})')

        evaluation_results.append(
            {
                'summary': {
                    'exact': findFirst(validation.exact.tops),
                    'exact-no-padding': findFirst(validation.exact_no_padding.tops),
                    'when-rule': findFirst(validation.when_rule.tops),
                    'with-padding': findFirst(validation.with_padding.tops)
                }
            }
        )
    con.commit()

    def hist(name):
        hist, bin_edges = np.histogram([(sample['summary'][name])
                                        for sample in evaluation_results], bins=list(range(21)))
        return {'hist': hist.tolist(), 'bin_edges': bin_edges.tolist()}

    histogram = {
        'exact': hist('exact'),
        'exact_no_padding': hist('exact-no-padding'),
        'when_rule': hist('when-rule'),
        'with_padding': hist('with-padding'),
    }

    return histogram


def main(config, options):
    setup_logging(**vars(options))

    root_path = Path(__file__).absolute().parents[2]
    dist_folder = root_path / 'dist/client'

    app = Flask(__name__, static_url_path=str(dist_folder))

    dataset = BagDataset.load(config.files.solver_trainings_data, data_size_limit=10 if options.smoke else None)
    scenario = Scenario.load(config.files.scenario)
    rule_mapping = get_rule_mapping(scenario)
    rule_id2verbose = {i: rule.verbose for i, rule in rule_mapping.items()}
    # rule_verbose2id = {rule.verbose: i for i, rule in rule_mapping.items()}
    inferencer = Inferencer(config, scenario, fresh_model=False)
    embed2ident = dataset.embed2ident
    assert dataset.ident_dict == inferencer.ident_dict, 'Inconsistent idents in model and dataset'

    histogram_data = create_index(inferencer, dataset)

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
        x, s, y, p, v, target, mask = dataset[index]
        py, pv = inferencer.inference(initial, keep_padding=True)
        py = np.transpose(py, (1, 0))
        target = np.transpose(target, (1, 0))
        mask = np.transpose(mask, (1, 0))

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
            'fitMask': mask.tolist(),
            'groundTruthValue': v.tolist()[0],
            'predictedValue': pv.tolist(),
            'predictedPolicy': py.tolist(),
            'gtPolicy': target.tolist(),
            'parts': [initial.latex_with_colors([(highlight_color, [])])] +
            [initial.latex_with_colors([off_focus, (highlight_color, path)]) for path in parts_path[1:]],
            'rules': [rule.latex_verbose for rule in dataset.get_rules_raw()],
            'possibleFits': possibleFits,
            'validationMetrics': validate(truth=y, predict=py).as_dict()
        })

    @app.route('/api/length')
    def sample_count():
        return jsonify({'length': len(dataset.container)})

    @app.route('/api/overview')
    def overview():
        begin = int(request.args['begin'])
        end = int(request.args['end'])
        sorting_key = request.args.get('key', 'none').lower()
        sorting_filter = request.args.get('filter', '')
        sorting_up = request.args.get('up', 'true')
        return jsonify(query(begin=begin, end=end,
                             up=sorting_up == 'true', sorting_key=sorting_key,
                             rule_filter=sorting_filter,
                             rule_id2verbose=rule_id2verbose))

    @app.route('/api/histogram')
    def histogram():
        return jsonify(histogram_data)

    @app.teardown_appcontext
    def close_connection(exception):
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    app.run()


if __name__ == "__main__":
    parser = ArgumentParser('-c', '--config-file', exclude="scenario-*",
                            prog='deep training')
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--smoke', action='store_true', default=False)
    main(*parser.parse_args())
