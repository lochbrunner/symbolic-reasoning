use core::Rule;
use core::Symbol;

pub struct ApplyInfo<'a> {
    pub rule: &'a Rule,
    pub initial: &'a Symbol,
    pub deduced: Symbol,
}

impl<'a> ApplyInfo<'a> {
    pub fn print_header() {
        println!("  {0: <10} | {1: <10} | {2: <10}", "new", "initial", "rule");
        println!("  -----------------------------------------");
    }

    pub fn print(&self) {
        let ded_str = format!("{}", self.deduced);
        let ini_str = format!("{}", self.initial);
        let rule = format!("{}", self.rule);
        println!("  {0: <10} | {1: <10} | {2: <10}", ded_str, ini_str, rule);
    }
}

pub struct Trace<'a> {
    pub initial: &'a Symbol,
    /// First stage
    pub stage: Vec<ApplyInfo<'a>>,
}
