#!/usr/bin/env python3

import logging
from glob import glob
from pathlib import Path
from time import time

import yaml

from pycore import Scenario, Symbol, Trace

from common import grid_search
from common import io
from common.config_and_arg_parser import ArgumentParser
from common.timer import Timer

from solver.beam_search import beam_search
from solver.inferencer import Inferencer
from solver.solve_problems import solve_problems
from solver.trace import ApplyInfo, solution_summary
from solver.trace import dump_new_rules


def solve_training_problems(training_traces, scenario, rule_mapping, trainings_data_max_steps, inferencer: Inferencer, **kwargs):

    def variable_generator():
        return Symbol.parse(scenario.declarations, 'u')

    failed = 0
    succeeded = 0
    total_duration = 0.
    seen = set()
    for filename in glob(training_traces):
        logging.info(f'Evaluating {filename} ...')
        trace = Trace.load(filename)
        for calculation in trace.unroll:
            conclusion = calculation.steps[0].initial
            steps_count = min(trainings_data_max_steps, len(calculation.steps)-1)
            condition = calculation.steps[steps_count].initial
            if condition in seen:
                continue
            seen.add(condition)
            start_time = time()
            solution, _ = beam_search(inferencer, rule_mapping, condition,
                                      [conclusion], variable_generator, **kwargs)
            total_duration += time() - start_time
            if solution is not None:
                succeeded += 1
            else:
                failed += 1

    logging.info(f'{succeeded} of {succeeded+failed} training traces succeeded')
    total = succeeded + failed
    return {'succeeded': succeeded, 'failed': failed, 'total': total, 'mean-duration': total_duration / total}


def print_solution(solution: ApplyInfo):
    for step in reversed(list(solution.trace)):
        if step.previous is not None:
            mapping = ', '.join([f'{a} -> {b}' for a, b in step.mapping.items()])
            if mapping:
                mapping = f' ({mapping})'
            print(f'{step.rule_name} ({step.rule_formula}): {step.previous.current.verbose} => {step.current.verbose}{mapping}')
        else:
            print(f'Initial: {step.current}')


def main(options, config):
    with Timer('Loading scenario'):
        scenario = Scenario.load(config.files.scenario)

    # model, idents, rules = io.load_model(model, depth=9)
    rules = io.load_rules(config.files.model)
    rule_mapping = {}
    used_rules = set()
    max_width = max(len(s.name) for s in scenario.rules.values())+1
    for i, model_rule in enumerate(rules[1:], 1):
        scenario_rule = next(rule for rule in scenario.rules.values() if rule.reverse.verbose == model_rule)
        rule_mapping[i] = scenario_rule
        used_rules.add(str(scenario_rule))
        logging.debug(f'Using rule {i:2}# {scenario_rule.name.ljust(max_width)} {scenario_rule.verbose}')

    for scenario_rule in scenario.rules.values():
        if str(scenario_rule) not in used_rules:
            logging.warning(f'The rule "{scenario_rule}" was not in the model created by the training.')

    inferencer = Inferencer(config=config, scenario=scenario, fresh_model=options.fresh_model, use_solver_data=True)

    training_statistics = []
    if options.solve_training:
        bs = [int(s) for s in str(config.evaluation.trainings_data.beam_size).split(':')]
        if len(bs) == 1:
            beam_size_range = bs
        else:
            beam_size_range = range(*bs)
        for bs in beam_size_range:
            logging.info(f'Using beam size {bs}')
            training_statistic = solve_training_problems(
                training_traces=config.files.trainings_data_traces, scenario=config.scenario,
                rule_mapping=rule_mapping, beam_size=bs,
                trainings_data_max_steps=config.evaluation.training_data.max_steps, inferencer=inferencer)
            training_statistic['beam-size'] = bs
            training_statistics.append(training_statistic)

    problem_solutions, problem_statistics = solve_problems(options, config, scenario, inferencer, rule_mapping)
    dump_new_rules(solutions=problem_solutions, new_rules_filename=config.evaluation.new_rules_filename)
    for problem_solution in problem_solutions:
        if problem_solution:
            print_solution(problem_solution)

    results_filename = Path(config.files.evaluation_results)
    logging.info(f'Writing results to {results_filename}')
    with results_filename.open('w') as f:
        yaml.dump({
            'problems': [d.as_dict() for d in problem_statistics],
            'problem-statistics': solution_summary(problem_solutions),
            'training-traces': training_statistics
        }, f)

    logging.info('Summary:')
    for problem_statistic in problem_statistics:
        success = 'success' if problem_statistic.success else 'failed'

        logging.info(f'{problem_statistic.name}: success {success} fits: {problem_statistic.fit_results}')


if __name__ == '__main__':
    parser = ArgumentParser(domain='evaluation', prog='solver', exclude='scenario-*')
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--results-filename')
    parser.add_argument('--policy-last', action='store_true', default=False)

    # Model
    parser.add_argument('--fresh-model', action='store_true', help='Creates a fresh model')

    config_args, self_args = parser.parse_args()
    loglevel = 'DEBUG' if self_args.verbose else self_args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )

    main(self_args, config_args)
