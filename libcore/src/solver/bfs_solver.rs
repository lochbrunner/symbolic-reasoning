use crate::fit::FitMap;
use crate::solver::statistics::{IntermediateStepInfo, SolveResult, Statistics};
use crate::{fit, Rule, Symbol};

pub struct Possibility<'a> {
    scenarios: Vec<FitMap<'a>>,
    rule: &'a Rule,
}

/// Offers heuristics for which rule to choose
pub trait Policy {
    fn choose(&self, prev: &Symbol, possibilities: &[Possibility]) -> Vec<f32>;
}

pub trait VariableGenerator<'a> {
    fn generate(&mut self) -> &'a Symbol;
}

/// Tries to solve the problem via BFS
/// It presents all possibilities and chooses one of them
pub fn present_and_solve<'a, P>(
    policy: &P,
    _variable_generator: &dyn VariableGenerator,
    timeout: u32,
    initial: &Symbol,
    _end: &Symbol,
    rules: &'a [Rule],
) -> SolveResult<'a>
where
    P: Policy,
{
    let mut stages = vec![vec![IntermediateStepInfo {
        deduced: initial.clone(),
        rule: None,
        prev_index: None,
        path: vec![],
    }]];

    let mut statistics = Statistics::new();
    let mut _seen = hashset! {initial.to_string()};
    // let variable_creator = &|| &variable_generator.generate();

    'stages: loop {
        for (_prev_index, prev_step) in stages.last().unwrap().iter().enumerate() {
            // Calculate all possibilities
            let possibilities = rules
                .iter()
                .map(|rule| Possibility {
                    rule,
                    scenarios: fit(&prev_step.deduced, &rule.condition),
                })
                .collect::<Vec<_>>();

            statistics.fit_calls_count += 1;
            let _dist = policy.choose(&prev_step.deduced, &possibilities);

            // TODO: Pick
            // TODO: Deduce

            if statistics.fit_calls_count > timeout {
                return SolveResult {
                    statistics,
                    trace: Err(()),
                };
            }
        }

        break 'stages;
    }

    SolveResult {
        statistics,
        trace: Err(()),
    }
}
