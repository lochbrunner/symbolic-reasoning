use core::{apply_batch, fit, Context, Rule, Symbol};
mod io;
use io::*;
mod variable_generator;
use variable_generator::*;

struct ApplyInfo<'a> {
    rule: &'a Rule,
    initial: &'a Symbol,
    deduced: Symbol,
}

impl<'a> ApplyInfo<'a> {
    fn print_header() {
        println!("  {0: <10} | {1: <10} | {2: <10}", "new", "initial", "rule");
        println!("  -----------------------------------------");
    }

    fn print(&self) {
        let ded_str = format!("{}", self.deduced);
        let ini_str = format!("{}", self.initial);
        let rule = format!("{}", self.rule);
        println!("  {0: <10} | {1: <10} | {2: <10}", ded_str, ini_str, rule);
    }
}

fn deduce_once<'a>(initial: &'a Symbol, rule: &'a Rule) -> Vec<ApplyInfo<'a>> {
    fit(initial, &rule.condition)
        .iter()
        .flat_map(|scenario| {
            apply_batch(
                scenario,
                variables_generator(initial), // May return vector
                initial,
                &rule.conclusion,
            )
        })
        .map(|deduced| ApplyInfo {
            rule,
            initial,
            deduced,
        })
        .collect()
}

fn deduce(initial: &Symbol, rules: &[Rule]) {
    let mut deduced: Vec<ApplyInfo> = Vec::new();

    for rule in rules.iter() {
        deduced.extend(deduce_once(initial, rule));
    }

    println!("Deduced:");
    ApplyInfo::print_header();
    for ded in deduced.iter() {
        ded.print();
    }
}

fn main() {
    let mut context = Context::load("./generator/assets/declarations.yaml");
    context.register_standard_operators();

    let rules = read_rules(&context, "./generator/assets/rules.txt");
    let premises = read_premises(&context, "./generator/assets/premises.txt");

    let initial = &premises[0];

    deduce(initial, &rules);
}
