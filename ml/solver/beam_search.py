
import logging

from solver.trace import ApplyInfo, Statistics
from pycore import fit_at_and_apply, fit_and_apply


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
                    confidence=confidence, top=top,
                    rule_id=rule_id, path=fit_result.path)

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
