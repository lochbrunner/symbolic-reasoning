use crate::iter_extensions::{PickTraitVec, Strategy};
use rose::draw_rose;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

extern crate clap;
use clap::{App, Arg};

mod variable_generator;
use variable_generator::*;
mod configuration;
mod iter_extensions;
mod rose;
mod svg;

use core::bag::trace::{ApplyInfo, DenseTrace, Meta, Trace, TraceStep};
use core::bag::Bag;
use core::scenario::Scenario;
use core::{apply_batch, fit, Context, Rule, Symbol};

#[macro_use]
extern crate serde_derive;

fn density(symbol: &Symbol) -> f32 {
    let spread: i32 = 2;
    let size = symbol.parts().map(|_| 1).sum::<i32>();
    let max_size = (0..symbol.depth).map(|i| spread.pow(i)).sum::<i32>();
    size as f32 / max_size as f32
}

fn gini(distribution: &[usize]) -> f64 {
    let sum = distribution.iter().sum::<usize>();
    let nom: isize = distribution
        .iter()
        .enumerate()
        .map(|(i, xi)| {
            distribution[i..]
                .iter()
                .map(|xj| (*xi as isize - *xj as isize).abs())
                .sum::<isize>()
        })
        .sum();
    nom as f64 / (2 * distribution.len() * sum) as f64
}

fn print_statistics(distribution: &[usize]) {
    let sum = distribution.iter().sum::<usize>();
    println!("sum: {}", sum);
    let min = distribution.iter().min().unwrap();
    println!("min: {}", min);
    println!("gini: {}", gini(distribution));
}

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

/// Filters out non sense symbols in the sene of e.g. a^0^0^0^0^0 ...
fn filter_interest<'a>(apply: &'a &ApplyInfo<'_>) -> bool {
    // Filter out when a pattern repeats directly
    for sub in apply.deduced.iter_bfs() {
        if sub.operator() {
            for child in sub.childs.iter() {
                if child.childs.len() == sub.childs.len() {
                    let a = sub.childs.iter().map(|s| &s.ident);
                    let b = child.childs.iter().map(|s| &s.ident);
                    if a.eq(b) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn deduce_impl<'a>(
    alphabet: &'a [Symbol],
    initial: &Symbol,
    rules: &'a [(String, Rule)],
    stages: &[usize],
    stage_index: usize,
    rules_distribution: &mut Vec<usize>,
) -> Vec<TraceStep<'a>> {
    if stages.len() == stage_index {
        return vec![];
    }
    let max_stage_size = stages[stage_index];
    let mut stage = vec![];
    // How to reduce for all rules in sum
    for (rule_id, (_, rule)) in rules.iter().enumerate() {
        let max_rule = rules_distribution
            .iter()
            .cloned()
            .fold(0, usize::max)
            .max(100) as usize;
        // max_stage_size*(1- rules_distribution[rule_id] / max_rule)^2
        let rel = 1. - rules_distribution[rule_id] as f64 / max_rule as f64;
        let stage_size = (max_stage_size as f64 * rel.powf(3.)) as usize;
        let new_calcs = deduce_once(alphabet, initial, rule)
            .pick(Strategy::Random(true))
            .filter(filter_interest)
            .take(stage_size)
            .cloned()
            .map(|a| TraceStep {
                successors: deduce_impl(
                    alphabet,
                    &a.deduced,
                    rules,
                    stages,
                    stage_index + 1,
                    rules_distribution,
                ),
                info: a,
            })
            .collect::<Vec<TraceStep>>();
        rules_distribution[rule_id] += new_calcs.len();
        stage.extend(new_calcs);
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
    rules: &'a [(String, Rule)],
    stages: &'a [usize],
) -> Trace<'a> {
    // Find all concrete ident of the rules
    let mut used_idents =
        extract_idents_from_rules(&rules.iter().map(|(_, r)| r.reverse()).collect::<Vec<_>>());

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

    let mut rules_distribution = vec![0; rules.len()];
    let trace = Trace {
        meta: Meta {
            used_idents,
            rules: rules.to_vec(),
        },
        initial,
        stages: deduce_impl(alphabet, initial, rules, stages, 0, &mut rules_distribution),
    };
    print_statistics(&rules_distribution);
    trace
}

fn main() {
    let matches = App::new("Sample data generator")
        .version("0.2.0")
        .author("Matthias Lochbrunner <matthias_lochbrunner@web.de>")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("configuration-filename")
                .help("file containing the configuration")
                .takes_value(true)
                .default_value("./real_world_problems/basics/generation.yaml"),
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

    let config_filename = matches.value_of("config").unwrap();
    let config = configuration::Configuration::load(config_filename).unwrap();

    let scenario = Scenario::load_from_yaml(&config.scenario).unwrap();

    let rules = scenario
        .rules
        .iter()
        .map(|(k, v)| (k.clone(), v.reverse()))
        .collect::<Vec<_>>();

    let postfix = config
        .stages
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let out_filename = format!("out/generator/bag-{}.bin", postfix);

    let alphabet = create_alphabet();
    let mut context = Context::load(&config.scenario)
        .unwrap_or_else(|_| panic!("Loading declarations from {}", &config.scenario));
    context.register_standard_operators();

    let traces = scenario
        .premises
        .iter()
        .map(|initial| deduce(&alphabet, initial, &rules, &config.stages))
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

    println!("Converting to bag");
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
