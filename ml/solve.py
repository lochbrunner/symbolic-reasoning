#!/usr/bin/env python3

import logging
import yaml
from glob import glob
from time import time
from pathlib import Path

from pycore import Symbol, Scenario, Trace, fit, apply, fit_at_and_apply, fit_and_apply

from common import io
from common.timer import Timer
from common.config_and_arg_parser import ArgumentParser
from common.parameter_search import LearningParmeter
from common import grid_search
from dataset import ScenarioParameter

from solver.inferencer import Inferencer
from solver.trace import ApplyInfo, Statistics, solution_summary, TrainingsDataDumper, dump_new_rules


def beam_search(inference, rule_mapping, initial, targets, variable_generator, beam_size, num_epochs, **kwargs):
    '''First apply the policy and then try to fit the suggestions.'''
    seen = set()
    statistics = Statistics(initial)

    for _ in range(num_epochs):
        for prev in statistics.trace:
            policies = inference(prev.current, beam_size)
            for top, (rule_id, path, confidence) in enumerate(policies, 1):
                rule = rule_mapping[rule_id-1]
                result = fit_at_and_apply(variable_generator, prev.current, rule, path)
                statistics.fit_tries += 1
                statistics.fit_results += 1 if result is not None else 0
                if result is None:
                    logging.debug(f'Missing fit of {rule.condition} at {path} in {prev.current}')
                    continue
                deduced, mapping = result
                s = deduced.verbose
                if s in seen:
                    continue
                seen.add(s)
                apply_info = ApplyInfo(
                    rule_name=rule.name, rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev, mapping=mapping,
                    confidence=confidence,
                    top=top,
                    rule_id=rule_id, path=path)
                statistics.trace.add(apply_info)
                if deduced in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics

        statistics.trace.close_stage()

    return None, statistics


def beam_search_policy_last(inference, rule_mapping, initial, targets, variable_generator, beam_size, num_epochs, black_list_terms, black_list_rules, max_size, **kwargs):
    '''Same as `beam_search` but first get fit results and then apply policy to sort the results.'''
    black_list_terms = set(black_list_terms)
    black_list_rules = set(black_list_rules)
    seen = set([initial.verbose])
    statistics = Statistics(initial)
    for epoch in range(num_epochs):
        logging.debug(f'epoch: {epoch}')
        successfull_epoch = False
        for prev in statistics.trace:
            possible_rules = {}
            for i, rule in rule_mapping.items():
                if rule.name not in black_list_rules:
                    if fits := fit_and_apply(variable_generator, prev.current, rule):
                        possible_rules[i] = fits

            # Sort the possible fits by the policy network
            policies, value = inference(prev.current, None)  # rule_id, path
            prev.value = value.item()
            ranked_fits = {}
            for rule_id, fits in possible_rules.items():
                for deduced, fit_result in fits:
                    try:
                        j, confidence = next((i, conf) for i, (pr, pp, conf) in enumerate(policies)
                                             if pr == rule_id and pp == fit_result.path)
                    except StopIteration:
                        for k, v in rule_mapping.items():
                            print(f'#{k}: {v}')
                        raise RuntimeError(f'Can not find {rule_mapping[rule_id]} #{rule_id} at {fit_result.path}')
                    # rule id, path, mapping, deduced
                    ranked_fits[j] = (rule_id, fit_result, confidence, deduced)

            possible_fits = (v for _, v in sorted(ranked_fits.items()))

            # filter out already seen terms
            possible_fits = ((*args, deduced) for *args, deduced in possible_fits if deduced.verbose
                             not in seen and deduced.verbose not in black_list_terms)

            for top, (rule_id, fit_result, confidence, deduced) in enumerate(possible_fits, 1):
                seen.add(deduced.verbose)
                if deduced.size > max_size:
                    continue
                statistics.fit_results += 1
                rule = rule_mapping[rule_id]
                # print(deduced.verbose)
                apply_info = ApplyInfo(
                    rule_name=rule.name, rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev, mapping=fit_result.variable,
                    confidence=confidence, top=top, rule_id=rule_id, path=fit_result.path)
                statistics.trace.add(apply_info)
                successfull_epoch = True

                if deduced in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics
        if not successfull_epoch:
            break
        statistics.trace.close_stage()

    return None, statistics


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


def main(self_args, config):
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

    inferencer = Inferencer(fresh_model=self_args.fresh_model, config=config, use_solver_data=True)

    training_statistics = []
    if self_args.solve_training:
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

    problem_statistics = []
    problem_solutions = []

    def variable_generator():
        return Symbol.parse(context, 'u')
    context = scenario.declarations

    trainings_data_dumper = TrainingsDataDumper(config)
    eval_config = config.evaluation

    for problem_name in scenario.problems:
        problem = scenario.problems[problem_name]
        # Get first problem
        logging.info(f'problem: {problem}')
        logging.info('-'*30)
        source = problem.condition
        target = problem.conclusion

        with Timer(f'Solving problem "{problem_name}"'):
            search_strategy = beam_search_policy_last if self_args.policy_last else beam_search
            solution, statistics = search_strategy(inferencer, rule_mapping, problem.condition,
                                                   [problem.conclusion], variable_generator,
                                                   beam_size=eval_config.problems.beam_size,
                                                   num_epochs=eval_config.problems.num_epochs,
                                                   black_list_terms=eval_config.black_list_terms,
                                                   black_list_rules=eval_config.black_list_rules,
                                                   max_size=eval_config.max_size)
        if solution is not None:
            problem_solutions.append(solution)
            for step in reversed(list(solution.trace)):
                if step.previous is not None:
                    mapping = ', '.join([f'{a} -> {b}' for a, b in step.mapping.items()])
                    if mapping:
                        mapping = f' ({mapping})'
                    print(f'{step.rule_name} ({step.rule_formula}): {step.previous.current.verbose} => {step.current.verbose}{mapping}')
                else:
                    print(f'Initial: {step.current}')

            trainings_data_dumper += statistics
        else:
            logging.warning(f'No solution found for {source} => {target}')

        statistics.name = problem_name
        logging.info(statistics)
        problem_statistics.append(statistics)

    dump_new_rules(solutions=problem_solutions, new_rules_filename=eval_config.new_rules_filename)
    trainings_data_dumper.dump()

    results_filename = Path(config.files.evaluation_results)
    logging.info(f'Writing results to {results_filename}')
    with results_filename.open('w') as f:
        yaml.dump({
            'problems': [d.as_dict() for d in problem_statistics],
            'problem-statistics': solution_summary(problem_solutions),
            'training-traces': training_statistics
        }, f)

    logging.info('Summary:')
    for problem_statistics in problem_statistics:
        success = 'success' if problem_statistics.success else 'failed'

        logging.info(f'{problem_statistics.name}: success {success} fits: {problem_statistics.fit_results}')


def load_config(filename):
    def unroll_nested_dict(dictionary, parent=''):
        for key, value in dictionary.items():
            if type(value) in (str, int, float, list):
                yield (f'{parent}{key}', value)
            elif type(value) is dict:
                for pair in unroll_nested_dict(value, f'{parent}{key}--'):
                    yield pair
            else:
                raise NotImplementedError(f'Loading config does not support type {type(value)}')

    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return {'model': config['files']['model'],
                'results-filename': config['files']['evaluation-results'],
                'training-traces': config['files']['trainings-data-traces'],
                'initial-trainings-data-file': config['files']['trainings-data'],
                'model-parameter': config['training']['model-parameter'],
                'training': config['training'],
                **{k: v for k, v in unroll_nested_dict(config['evaluation'])}
                }


def get_learning_params(args, model_hyper_parameters, defaults):
    # get trainings parameter
    # trainings_param = dict((n.replace('training__', ''), v)
    #                        for n, v in vars(args).items() if n.startswith('training_'))
    trainings_param = defaults['training']
    model_hyper_parameter, arg = next(grid_search.unroll_many(model_hyper_parameters, vars(trainings_param)))
    learn_params = LearningParmeter(model_hyper_parameter=model_hyper_parameter, **arg)
    scenario_params = ScenarioParameter(**arg)
    return learn_params, scenario_params


if __name__ == '__main__':
    parser = ArgumentParser(domain='evaluation', prog='solver', exclude="scenario-*")
    # Common
    parser.add_argument('--log', help='Set the log level', default='warning')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--solve-training', help='Tries to solve the trainings data', action='store_true')
    parser.add_argument('--results-filename')
    parser.add_argument('--policy-last', action='store_true', default=False)

    # Model
    parser.add_argument('--fresh-model', help='Creates a fresh model')

    config_args, self_args = parser.parse_args()
    loglevel = 'DEBUG' if self_args.verbose else self_args.log.upper()

    logging.basicConfig(
        level=logging.getLevelName(loglevel),
        format='%(message)s'
    )

    main(self_args, config_args)
