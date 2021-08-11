use crate::{Rule, Symbol};

pub struct IntermediateStepInfo<'a> {
    pub rule: Option<&'a Rule>,
    pub path: Vec<usize>,
    pub prev_index: Option<usize>,
    pub deduced: Symbol,
}

#[derive(Debug)]
pub struct CalculationStep<'a> {
    pub rule: &'a Rule,
    pub path: Vec<usize>,
    pub term: Symbol,
}

pub struct Statistics {
    pub fits_count: u32,
    pub applies_count: u32,
    pub fit_calls_count: u32,
}

impl Statistics {
    pub fn new() -> Statistics {
        Statistics {
            fits_count: 0,
            applies_count: 0,
            fit_calls_count: 0,
        }
    }
}

pub struct SolveResult<'a> {
    pub trace: Result<Vec<CalculationStep<'a>>, ()>,
    pub statistics: Statistics,
}
