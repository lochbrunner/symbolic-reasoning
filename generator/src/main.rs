use crate::iter_extensions::{PickTraitVec, Strategy};
use rose::draw_rose;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

mod variable_generator;
use variable_generator::*;
mod iter_extensions;
mod rose;
mod svg;

use core::bag::trace::{ApplyInfo, DenseTrace, Meta, Trace, TraceStep};
use core::bag::Bag;
use core::scenario::Scenario;
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
    let mut used_idents =
        extract_idents_from_rules(&rules.iter().map(|r| r.reverse()).collect::<Vec<_>>());

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

    Trace {
        meta: Meta {
            used_idents,
            rules: rules.to_vec(),
        },
        initial,
        stages: deduce_impl(alphabet, initial, rules, stages, 0),
    }
}

fn main() {
    let matches = App::new("Sample data generator")
        .version("0.2.0")
        .author("Matthias Lochbrunner <matthias_lochbrunner@web.de>")
        .arg(
            Arg::with_name("stages")
                .long("stages")
                .help("numbers of fits (per rule) to use for each stage")
                .multiple(true)
                .takes_value(true)
                .default_value("1"),
        )
        .arg(
            Arg::with_name("scenario")
                .short("s")
                .long("scenario-filename")
                .help("file containing the declarations")
                .takes_value(true)
                .default_value("./real_world_problems/basics/dataset.yaml"),
        )
        .arg(
            Arg::with_name("rose")
                .long("rose-directory")
                .help("Directory to place the roses")
                .takes_value(true)
                .default_value("./out/generator/rose"),
        )
        .arg(
            Arg::with_name("tex")
                .long("tex-directory")
                .help("Directory to place the tex files")
                .takes_value(true),
        )
        .get_matches();

    let stages = values_t!(matches, "stages", usize)
        .unwrap()
        .into_iter()
        .collect::<Vec<usize>>();
    let declaration_filename = matches.value_of("scenario").unwrap();
    let scenario = Scenario::load_from_yaml(declaration_filename).unwrap();

    let rules = scenario
        .rules
        .iter()
        .map(|(_, v)| v.reverse())
        .collect::<Vec<_>>();

    let postfix = stages
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let out_filename = format!("out/generator/bag-{}.bin", postfix);

    let alphabet = create_alphabet();
    let mut context = Context::load(declaration_filename)
        .unwrap_or_else(|_| panic!("Loading declarations from {}", declaration_filename));
    context.register_standard_operators();

    let traces = scenario
        .premises
        .iter()
        .map(|initial| deduce(&alphabet, initial, &rules, &stages))
        .collect::<Vec<_>>();

    if let Some(dir) = matches.value_of("rose") {
        if !Path::new(dir).is_dir() {
            println!("\"{}\" is not a valid directory!", dir);
            return;
        }
        for (i, trace) in traces.iter().enumerate() {
            let filename = Path::new(dir).join(&format!("rose_{}.svg", i));
            let filename = filename.to_str().unwrap();
            draw_rose(filename, trace).expect("SVG Dump");
        }
    }

    let bag = Bag::from_traces(&traces);

    println!("Writing bag file to \"{}\" ...", out_filename);
    let writer = BufWriter::new(File::create(out_filename).unwrap());
    bag.write_bincode(writer).expect("Writing bin file");

    if let Some(dir) = matches.value_of("tex") {
        if !Path::new(dir).is_dir() {
            println!("\"{}\" is not a valid directory!", dir);
            return;
        }
        for (i, trace) in traces.iter().enumerate() {
            let filename = Path::new(dir).join(&format!("trace_{}.tex", i));
            let mut writer = BufWriter::new(File::create(filename).expect("Parent folder exists"));
            let trace = DenseTrace::from_trace(trace);
            trace.write_latex(&mut writer).expect("Writing tex file");
        }
    }
}
