use crate::iter_extensions::{PickTraitVec, Strategy};
use crate::trace::{ApplyInfo, Trace, TraceStep};
use core::{apply_batch, fit, Context, Rule, Symbol};
use rose::draw_rose;
mod io;
use io::*;
mod variable_generator;
use variable_generator::*;
mod iter_extensions;
mod rose;
mod svg;
mod trace;

fn deduce_once<'a>(alphabet: &[Symbol], initial: &Symbol, rule: &'a Rule) -> Vec<ApplyInfo<'a>> {
    fit(initial, &rule.condition)
        .iter()
        .flat_map(|scenario| {
            apply_batch(
                scenario,
                variables_generator(initial, &alphabet),
                initial,
                &rule.conclusion,
            )
            .into_iter()
            .map(|r| (r, scenario.path.clone()))
            .collect::<Vec<(Symbol, Vec<usize>)>>()
        })
        .map(|deduced| ApplyInfo {
            rule,
            initial: initial.clone(),
            deduced: deduced.0,
            path: deduced.1,
        })
        .collect()
}

fn deduce_impl<'a>(
    alphabet: &'a [Symbol],
    initial: &Symbol,
    rules: &'a [Rule],
    stages: &[usize],
    stage_index: usize,
) -> Vec<TraceStep<'a>> {
    if stages.len() == stage_index {
        return vec![];
    }
    let mut stage = vec![];
    // How to reduce for all rules in sum
    for rule in rules.iter() {
        stage.extend(
            deduce_once(alphabet, initial, rule)
                .pick(Strategy::Random(true))
                .take(stages[stage_index])
                .cloned()
                .map(|a| TraceStep {
                    successors: deduce_impl(alphabet, &a.deduced, rules, stages, stage_index + 1),
                    info: a,
                })
                .collect::<Vec<TraceStep>>(),
        );
    }
    stage
}

fn deduce(initial: &Symbol, rules: &[Rule], stages: Vec<usize>) {
    let alphabet = create_alphabet();

    let trace = Trace {
        initial,
        stage: deduce_impl(&alphabet, initial, rules, &stages, 0),
    };

    draw_rose("./out/generator/deduced.svg", &trace).expect("SVG Dump");
}

fn main() {
    let mut context = Context::load("./generator/assets/declarations.yaml");
    context.register_standard_operators();

    let rules = read_rules(&context, "./generator/assets/rules.txt");
    let premises = read_premises(&context, "./generator/assets/premises.txt");

    let initial = &premises[0];

    deduce(initial, &rules, vec![2, 2, 2]);
}
