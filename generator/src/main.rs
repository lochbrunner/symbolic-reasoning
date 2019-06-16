use crate::iter_extensions::{PickTraitVec, Strategy};
use rose::draw_rose;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

mod variable_generator;
use variable_generator::*;
mod iter_extensions;
mod rose;
mod svg;

use core::bag::trace::{ApplyInfo, Meta, Trace, TraceStep};
use core::bag::Bag;
use core::io::*;
use core::{apply_batch, fit, Context, Rule, Symbol};

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

fn extract_idents_from_rules(rules: &[Rule]) -> HashSet<String> {
    let mut used_idents = HashSet::new();

    for rule in rules.iter() {
        for part in rule
            .condition
            .parts()
            .filter(|s| s.fixed())
            .map(|s| &s.ident)
        {
            if !used_idents.contains(part) {
                used_idents.insert(part.clone());
            }
        }
    }

    used_idents
}

fn deduce<'a>(
    alphabet: &'a [Symbol],
    initial: &'a Symbol,
    rules: &'a [Rule],
    stages: &'a [usize],
) -> Trace<'a> {
    // Find all concrete ident of the rules
    let mut used_idents = extract_idents_from_rules(rules);

    for part in initial.parts() {
        if !used_idents.contains(&part.ident) {
            used_idents.insert(part.ident.clone());
        }
    }

    for item in alphabet.iter() {
        for part in item.parts() {
            if !used_idents.contains(&part.ident) {
                used_idents.insert(part.ident.clone());
            }
        }
    }

    let trace = Trace {
        meta: Meta {
            used_idents,
            rules: rules.to_vec(),
        },
        initial,
        stages: deduce_impl(alphabet, initial, rules, stages, 0),
    };

    draw_rose("./out/generator/deduced.svg", &trace).expect("SVG Dump");
    trace
}

fn main() {
    let matches = App::new("Sample data generator")
        .version("0.1.0")
        .author("Matthias Lochbrunner <matthias_lochbrunner@web.de>")
        .arg(
            Arg::with_name("stages")
                .short("s")
                .long("stages")
                .help("numbers of fits (per rule) to use for each stage")
                .multiple(true)
                .takes_value(true)
                .default_value("1"),
        )
        .arg(
            Arg::with_name("declaration")
                .short("d")
                .long("declaration-filename")
                .help("file containing the declarations")
                .takes_value(true)
                .default_value("./generator/assets/declarations.yaml"),
        )
        .arg(
            Arg::with_name("rules")
                .short("r")
                .long("rules-filename")
                .help("file containing the rules")
                .takes_value(true)
                .default_value("./generator/assets/rules.txt"),
        )
        .arg(
            Arg::with_name("premises")
                .short("p")
                .long("premises-filename")
                .help("file containing the premises")
                .takes_value(true)
                .default_value("./generator/assets/premises.txt"),
        )
        .get_matches();

    let stages = values_t!(matches, "stages", usize)
        .unwrap()
        .into_iter()
        .collect::<Vec<usize>>();
    let declaration_filename = matches.value_of("declaration").unwrap();
    let rules_filename = matches.value_of("rules").unwrap();
    let premises_filename = matches.value_of("premises").unwrap();
    let postfix = stages
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let out_filename = format!("out/generator/bag-{}.bin", postfix);

    let alphabet = create_alphabet();
    let mut context = Context::load(declaration_filename).expect(&format!(
        "Loading declarations from {}",
        declaration_filename
    ));
    context.register_standard_operators();

    let rules = read_rules(&context, rules_filename, Mode::Reversed);
    let premises = read_premises(&context, premises_filename);

    let traces = premises
        .iter()
        .map(|initial| deduce(&alphabet, initial, &rules, &stages))
        .collect::<Vec<_>>();

    let bag = Bag::from_traces(&traces);

    let writer = BufWriter::new(File::create(out_filename).unwrap());
    bag.write_bincode(writer).expect("Writing bin file");

    // let mut writer = BufWriter::new(File::create("out/generator/trace.tex").unwrap());
    // trace.write_latex(&mut writer).expect("Writing tex file");
}
