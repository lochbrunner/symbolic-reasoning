use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{create_dir_all, File};
use std::io::BufWriter;
use std::path::Path;
use std::vec;

extern crate clap;
use clap::{App, Arg};

mod augmentation;
mod configuration;
mod filter;
mod iter_extensions;
mod rose;
mod svg;
mod variable_generator;

use crate::augmentation::{augment_with_permuted_free_idents, AugmentationStrategy};
use crate::configuration::{Configuration, ConfigurationOverrides};
use crate::filter::{filter_interest, filter_out_blacklist, filter_out_repeating_patterns};
use crate::iter_extensions::{PickTraitVec, Strategy};
use crate::rose::draw_rose;
use crate::variable_generator::*;

use core::bag::trace::{ApplyInfo, DenseTrace, Meta, Trace, TraceStep};
use core::bag::{Bag, FitInfo};
use core::scenario::Scenario;
use core::{apply_batch, fit, Context, Rule, Symbol};

#[macro_use]
extern crate serde_derive;

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

fn print_statistics(distribution: &[usize], label: &str) {
    let sum = distribution.iter().sum::<usize>();
    let min = distribution.iter().min().unwrap();
    println!(
        "{}  total: {} min: {} gini: {:.3}",
        label,
        sum,
        min,
        gini(distribution)
    );
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

fn deduce_impl<'a>(
    config: &Configuration,
    alphabet: &'a [Symbol],
    initial: &Symbol,
    rules: &'a [(String, Rule)],
    stages: &[usize],
    stage_index: usize,
    rules_distribution: &mut Vec<usize>,
    prev_symbols: &[u64],
) -> Vec<TraceStep<'a>> {
    if stages.len() == stage_index {
        return vec![];
    }
    let soft_min = |min: f32, value: f32| -> bool {
        let progress = stage_index as f32 / stages.len() as f32;
        value >= min * progress
    };
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
        let stage_size =
            (max_stage_size as f64 * rel.powf(config.distribution_suppression_exponent)) as usize;
        let new_calcs = deduce_once(alphabet, initial, rule)
            .pick(Strategy::Random(true))
            .filter(|a| a.deduced.depth <= config.max_depth)
            .filter(|a| soft_min(config.min_working_density, a.deduced.density()))
            .filter(|a| filter_out_blacklist(config, a))
            .filter(filter_interest)
            .filter(filter_out_repeating_patterns)
            .filter(|a| !prev_symbols.iter().any(|p| *p == a.deduced.get_hash()))
            .take(stage_size)
            .cloned()
            .map(|a| TraceStep {
                successors: deduce_impl(
                    config,
                    alphabet,
                    &a.deduced,
                    rules,
                    stages,
                    stage_index + 1,
                    rules_distribution,
                    &([&prev_symbols[..], &[a.deduced.get_hash()]]).concat(),
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
    config: &Configuration,
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
        stages: deduce_impl(
            config,
            alphabet,
            initial,
            rules,
            stages,
            0,
            &mut rules_distribution,
            &[initial.get_hash()],
        ),
    };
    print_statistics(&rules_distribution, &format!("{}", initial));
    trace
}

fn create_parent_dir(filename: &str) {
    let directory = Path::new(filename).parent().unwrap();
    if !directory.is_dir() {
        println!(
            "Creating directory \"{}\" as it does not exist yet",
            directory.display()
        );
        create_dir_all(directory).unwrap();
    }
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
                .default_value("./real_world_problems/basics/dataset.yaml"),
        )
        .arg(
            Arg::with_name("rose")
                .long("rose-directory")
                .help("Directory to place the roses")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("dump_trace")
                .long("dump-trace")
                .help("Dumps the trace"),
        )
        .arg(
            Arg::with_name("tex")
                .long("tex-directory")
                .help("Directory to place the tex files")
                .takes_value(true),
        )
        .config_overrides()
        .get_matches();

    let config_filename = matches.value_of("config").unwrap();
    let config = Configuration::load(config_filename, &matches)
        .expect(&format!("load config {}", config_filename));

    let scenario = Scenario::load_from_yaml(&config_filename).unwrap();

    let rules = scenario
        .rules
        .iter()
        .map(|(k, v)| (k.clone(), v.reverse()))
        .collect::<Vec<_>>();

    let alphabet = create_alphabet();
    let mut context = Context::load(&config_filename)
        .unwrap_or_else(|_| panic!("Loading declarations from {}", &config_filename));
    context.register_standard_operators();

    if scenario.premises.is_empty() {
        println!("Warning: No premises found in the configuration!");
        let bag = Bag::empty(2, rules.clone());

        println!(
            "Writing empty bag file to \"{}\" ...",
            &config.dump_filename
        );
        create_parent_dir(&config.dump_filename);
        let writer = BufWriter::new(File::create(&config.dump_filename).unwrap());
        bag.write_bincode(writer).expect("Writing bin file");
        std::process::exit(0);
    }

    let traces = (&scenario.premises)[..]
        .par_iter()
        .map(|initial| deduce(&config, &alphabet, initial, &rules, &config.stages))
        .collect::<Vec<_>>();

    println!("Converting to bag...");
    let mut leafs: HashSet<&Symbol> = HashSet::new();
    for trace in traces.iter() {
        for step in trace.all_steps() {
            for sub in step.deduced.iter_bfs() {
                if sub.childs.is_empty() && !leafs.contains(sub) {
                    leafs.insert(sub);
                }
            }
        }
    }
    let leafs = leafs.into_iter().collect::<Vec<_>>();
    let mut idents: HashSet<&str> = HashSet::new();
    for trace in traces.iter() {
        for ident in trace.meta.used_idents.iter() {
            idents.insert(ident);
        }
    }
    // Find all free idents
    let free_idents = idents
        .into_iter()
        .filter(|s| s.len() == 1 && s.chars().next().unwrap().is_alphabetic())
        .collect::<HashSet<_>>();
    let augmentation = &|s: (&Symbol, FitInfo)| {
        if config.augmentation.enabled {
            augment_with_permuted_free_idents(
                &free_idents,
                &leafs,
                AugmentationStrategy::Random(config.augmentation.factor),
                s,
            )
        } else {
            let (symbol, fitinfo) = s;
            vec![(symbol.clone(), fitinfo)]
        }
    };
    let bag = Bag::from_traces(
        &traces,
        &|s| s.density() >= config.min_result_density && s.size() <= config.max_size,
        augmentation,
    );

    println!("Writing bag file to \"{}\" ...", &config.dump_filename);
    create_parent_dir(&config.dump_filename);
    let writer = BufWriter::new(File::create(&config.dump_filename).unwrap());
    bag.write_bincode(writer).expect("Writing bin file");

    if matches.is_present("dump_trace") {
        println!("Creating dense traces ...");
        let dense_traces = traces
            .iter()
            .map(DenseTrace::from_trace)
            .collect::<Vec<_>>();
        for (i, trace) in dense_traces.iter().enumerate() {
            let filename = config.trace_filename.replace('*', &i.to_string());
            println!("Writing trace file to \"{}\" ...", filename);
            let writer = BufWriter::new(File::create(filename).unwrap());
            trace.write_bincode(writer).expect("Writing trace file");
        }
    }

    if let Some(dir) = matches.value_of("tex") {
        if !Path::new(dir).is_dir() {
            eprintln!("\"{}\" is not a valid directory!", dir);
            std::process::exit(1);
        }
        for (i, trace) in traces.iter().enumerate() {
            let filename = Path::new(dir).join(&format!("trace_{}.tex", i));
            let mut writer = BufWriter::new(File::create(filename).expect("Parent folder exists"));
            let trace = DenseTrace::from_trace(trace);
            trace.write_latex(&mut writer).expect("Writing tex file");
        }
    }

    if let Some(dir) = matches.value_of("rose") {
        if !Path::new(dir).is_dir() {
            eprintln!("\"{}\" is not a valid directory!", dir);
            std::process::exit(1);
        }
        for (i, trace) in traces.iter().enumerate() {
            let filename = Path::new(dir).join(&format!("rose_{}.svg", i));
            let filename = filename.to_str().unwrap();
            draw_rose(filename, trace).expect("SVG Dump");
        }
    }
}
