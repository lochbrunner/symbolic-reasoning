import logging
from typing import Any, Callable
from pycore import Rule

# from memory_profiler import profile
import random

from solver.inferencer import Inferencer
from solver.trace import ApplyInfo, Statistics
from pycore import fit_at_and_apply, fit_and_apply

logger = logging.getLogger(__name__)


def beam_search(
    inference: Inferencer,
    rule_mapping: dict[int, Rule],
    initial,
    targets,
    variable_generator: Callable,
    num_epochs: int,
    beam_size: int,
    black_list_terms,
    black_list_rules,
    **kwargs,
):
    '''First apply the policy and then try to fit the suggestions.'''
    seen = set()
    black_list_terms = set(black_list_terms)
    black_list_rules = set(black_list_rules)
    statistics = Statistics(initial)

    for epoch in range(num_epochs):
        logger.debug(f'epoch: {epoch}')
        for prev in statistics.trace:
            policies, _ = inference(prev.current, beam_size)

            for top, (rule_id, path, confidence) in enumerate(policies, 1):
                rule = rule_mapping[rule_id - 1]
                if rule.name in black_list_rules:
                    continue
                result = fit_at_and_apply(variable_generator, prev.current, rule, path)
                statistics.fit_tries += 1
                statistics.fit_results += 1 if result is not None else 0
                if result is None:
                    logger.debug(
                        f'Missing fit of {rule.condition} at {path} in {prev.current}'
                    )
                    continue
                deduced, mapping = result
                s = deduced.verbose
                if s in seen or s in black_list_terms:
                    continue
                seen.add(s)
                apply_info = ApplyInfo(
                    rule_name=rule.name,
                    rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev,
                    mapping=mapping,
                    confidence=confidence,
                    top=top,
                    rule_id=rule_id,
                    path=path,
                )
                statistics.trace.add(apply_info)
                if deduced in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics

        statistics.trace.close_stage()

    return None, statistics


# @profile
def beam_search_policy_last(
    *,
    inference: Inferencer,
    rule_mapping,
    initial,
    targets,
    variable_generator,
    num_epochs: int,
    beam_size: int,
    max_track_loss: int,
    black_list_terms: list[str],
    white_list_terms: list[str],
    black_list_rules: list[str],
    max_size: int,
    max_grow: int,
    max_fit_results: int,
    use_network=True,
    exploration_ratio: int,
    shuffle_fits=True,
    **kwargs,
):
    '''Same as `beam_search` but first get fit results and then apply policy to sort the results.'''

    if not use_network:
        logger.debug('Don\'t use policy and value network. Just try brutforce solving.')
    if exploration_ratio is None or exploration_ratio == 0:
        exploration_ratio = num_epochs + 1

    black_list_terms = (black_list_terms or None) and set(black_list_terms)
    black_list_rules = set(black_list_rules)
    white_list_terms = set(white_list_terms)
    seen = {initial.verbose: None}
    statistics = Statistics(initial)
    max_size = min(max_size, initial.size + max_grow)

    targets = set(t.verbose for t in targets)

    # print(f'initial: {initial}')
    # print(f'targets: {targets}')
    # print(f'num_epochs: {num_epochs}')
    # print(f'white_list_terms: {white_list_terms}')

    for epoch in range(num_epochs):
        logger.debug(f'epoch: {epoch}')
        # print(f'epoch: {epoch}')
        successful_epoch = False
        for prev in statistics.trace:
            possible_rules = {}
            for i, rule in rule_mapping.items():
                if rule.name not in black_list_rules:
                    if fits := fit_and_apply(variable_generator, prev.current, rule):
                        possible_rules[i] = fits

            # Apply network outcome
            if use_network and random.random() > exploration_ratio:
                # Sort the possible fits by the policy network
                policies, value = inference(prev.current, None)  # rule_id, path
                prev.value = value.item()
                ranked_fits = {}
                for rule_id, fits in possible_rules.items():
                    for deduced, fit_result in fits:
                        try:
                            j, confidence = next(
                                (i, conf)
                                for i, (p_rule_id, p_path, conf) in enumerate(policies)
                                if p_rule_id == rule_id and p_path == fit_result.path
                            )
                        except StopIteration:
                            print('Available rules:')
                            for k, v in rule_mapping.items():
                                print(f'#{k}: {v}')
                            raise RuntimeError(
                                f'Can not find {rule_mapping[rule_id]} #{rule_id} at {fit_result.path}.'
                            )
                        # rule id, path, mapping, deduced
                        ranked_fits[j] = (rule_id, fit_result, confidence, deduced)

                possible_fits = (v for _, v in sorted(ranked_fits.items()))

                # filter out already seen terms
                possible_fits = [
                    (*args, deduced)
                    for *args, deduced in possible_fits
                    if deduced.verbose not in black_list_terms
                ]
                if beam_size is not None:
                    possible_fits = possible_fits[:beam_size]
            else:
                possible_fits = []
                for rule_id, fits in possible_rules.items():
                    for deduced, fit_result in fits:
                        confidence = None
                        possible_fits.append((rule_id, fit_result, confidence, deduced))

                if shuffle_fits:
                    random.shuffle(possible_fits)

                if beam_size is not None:
                    possible_fits = possible_fits[:beam_size]

            for top, (rule_id, fit_result, confidence, deduced) in enumerate(
                possible_fits, 1
            ):
                if deduced.size > max_size:
                    continue

                if white_list_terms and not deduced.verbose in white_list_terms:
                    continue

                statistics.fit_results += 1
                rule = rule_mapping[rule_id]
                apply_info = ApplyInfo(
                    rule_name=rule.name,
                    rule_formula=rule.verbose,
                    current=deduced,
                    previous=prev,
                    mapping=fit_result.variable,
                    confidence=confidence,
                    top=top,
                    rule_id=rule_id,
                    path=fit_result.path,
                )

                # Loop detection
                if deduced.verbose in seen:
                    # Initial is None
                    if (sister := seen[deduced.verbose]) is not None:
                        sister.alternative_traces.append(apply_info)
                    continue
                seen[deduced.verbose] = apply_info

                if use_network and apply_info.track_loss > max_track_loss:
                    continue

                statistics.trace.add(apply_info)
                successful_epoch = True

                if deduced.verbose in targets:
                    statistics.success = True
                    apply_info.contribute()
                    return apply_info, statistics

                if statistics.fit_results >= max_fit_results:
                    return None, statistics

        if not successful_epoch:
            break
        statistics.trace.close_stage()

    return None, statistics
