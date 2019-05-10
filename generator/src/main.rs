use crate::trace::ApplyInfo;
use crate::trace::Trace;
use core::{apply_batch, fit, Context, Rule, Symbol};
use rose::draw_rose;
mod io;
use io::*;
mod variable_generator;
use variable_generator::*;
mod rose;
mod svg;
mod trace;

fn deduce_once<'a>(initial: &'a Symbol, rule: &'a Rule) -> Vec<ApplyInfo<'a>> {
    let alphabet = create_alphabet();
    fit(initial, &rule.condition)
        .iter()
        .flat_map(|scenario| {
            apply_batch(
                scenario,
                variables_generator(initial, &alphabet), // May return vector
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
    let mut trace = Trace {
        initial,
        stage: Vec::new(),
    };

    for rule in rules.iter() {
        trace.stage.extend(deduce_once(initial, rule));
    }

    println!("Deduced:");
    ApplyInfo::print_header();
    for ded in trace.stage.iter() {
        ded.print();
    }

    draw_rose("./out/generator/deduced.svg", &trace).expect("SVG Dump");
}

fn main() {
    let mut context = Context::load("./generator/assets/declarations.yaml");
    context.register_standard_operators();

    let rules = read_rules(&context, "./generator/assets/rules.txt");
    let premises = read_premises(&context, "./generator/assets/premises.txt");

    let initial = &premises[0];

    deduce(initial, &rules);
}
