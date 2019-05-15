use core::Rule;
use core::Symbol;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ApplyInfo<'a> {
    pub rule: &'a Rule,
    pub path: Vec<usize>,
    pub initial: Symbol,
    pub deduced: Symbol,
}

impl<'a> ApplyInfo<'a> {
    #[allow(dead_code)]
    pub fn print_header() {
        println!("  {0: <14} | {1: <14} | {2: <14}", "new", "initial", "rule");
        println!("  -----------------------------------------");
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        let ded_str = format!("{}", self.deduced);
        let ini_str = format!("{}", self.initial);
        let rule = format!("{}", self.rule);
        println!("  {0: <14} | {1: <14} | {2: <14}", ded_str, ini_str, rule);
    }
}

#[derive(PartialEq, Eq, Hash)]
pub struct TraceStep<'a> {
    pub info: ApplyInfo<'a>,
    pub successors: Vec<TraceStep<'a>>,
}

pub struct Trace<'a> {
    pub initial: &'a Symbol,
    /// First stage
    pub stage: Vec<TraceStep<'a>>,
}
