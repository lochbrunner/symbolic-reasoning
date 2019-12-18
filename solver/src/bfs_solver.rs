use core::{apply, fit, solver::*, Rule, Symbol};

fn create_trace<'a>(
    index: usize,
    stages: &[Vec<IntermediateStepInfo<'a>>],
) -> Vec<CalculationStep<'a>> {
    let mut trace = vec![];
    let mut index = index;
    for stage in stages.iter().rev() {
        let info = &stage[index];
        match info.rule {
            None => (),
            Some(rule) => {
                trace.push(CalculationStep {
                    rule,
                    path: info.path.clone(),
                    term: info.deduced.clone(),
                });
            }
        }
        index = info.prev_index.unwrap_or(0);
    }
    trace.reverse();
    trace
}

/// Breadth-force search
pub fn solve<'a>(
    initial: &Symbol,
    end: &Symbol,
    rules: &'a [Rule],
    timeout: u32,
) -> SolveResult<'a> {
    let mut seen = hashset! {initial.to_string()};
    let def_symbol = Symbol::new_variable("a", false);

    let mut stages = vec![vec![IntermediateStepInfo {
        deduced: initial.clone(),
        rule: None,
        prev_index: None,
        path: vec![],
    }]];
    let mut statistics = Statistics::new();
    let variable_creator = &|| &def_symbol;
    loop {
        let mut next = vec![];
        for rule in rules.iter() {
            for (prev_index, prev_step) in stages.last().unwrap().iter().enumerate() {
                statistics.fit_calls_count += 1;
                if statistics.fit_calls_count > timeout {
                    return SolveResult {
                        statistics,
                        trace: Err(()),
                    };
                }
                let matches = fit(&prev_step.deduced, &rule.condition);
                statistics.fits_count += 1;
                for m in matches.iter() {
                    statistics.applies_count += 1;
                    let deduced = apply(m, variable_creator, &prev_step.deduced, &rule.conclusion);

                    let hash = deduced.to_string();
                    if !seen.contains(&hash) {
                        seen.insert(hash);

                        let finished = deduced == *end;
                        let step = IntermediateStepInfo {
                            rule: Some(rule),
                            path: m.path.clone(),
                            prev_index: Some(prev_index),
                            deduced,
                        };
                        next.push(step);

                        if finished {
                            stages.push(next);
                            return SolveResult {
                                statistics,
                                trace: Ok(create_trace(prev_index, &stages)),
                            };
                        }
                    }
                }
            }
        }
        stages.push(next);
    }
}
