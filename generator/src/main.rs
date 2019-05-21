use crate::iter_extensions::{PickTraitVec, Strategy};
use core::DenseTrace;
use core::{apply_batch, fit, ApplyInfo, Context, Rule, Symbol, Trace, TraceStep};
use rose::draw_rose;
mod io;
use io::*;
use std::fs::File;
use std::io::BufWriter;
mod variable_generator;
use variable_generator::*;
mod iter_extensions;
mod rose;
mod svg;

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

fn deduce(initial: &Symbol, rules: &[Rule], stages: Vec<usize>) -> DenseTrace {
    let alphabet = create_alphabet();

    let trace = Trace {
        initial,
        stage: deduce_impl(&alphabet, initial, rules, &stages, 0),
    };

    draw_rose("./out/generator/deduced.svg", &trace).expect("SVG Dump");

    DenseTrace::from_trace(&trace)
}

fn main() {
    let mut context = Context::load("./generator/assets/declarations.yaml");
    context.register_standard_operators();

    let rules = read_rules(&context, "./generator/assets/rules.txt");
    let premises = read_premises(&context, "./generator/assets/premises.txt");

    let initial = &premises[0];

    let trace = deduce(initial, &rules, vec![1, 1, 1]);

    let writer = BufWriter::new(File::create("out/trace.yaml").unwrap());
    trace.write_yaml(writer).expect("Writing.yaml file");

    let writer = BufWriter::new(File::create("out/trace.bin").unwrap());
    trace.write_bincode(writer).expect("Writing.bin file");

    // let reader = BufReader::new(File::open("out/trace.bin").expect("Opening trace.bin"));
    // let trace_loaded = DenseTrace::read_bincode(reader);

    let mut writer = BufWriter::new(File::create("out/trace.tex").unwrap());
    trace.write_latex(&mut writer).expect("Writing.bin file");
}
