use core::io;
use core::{Context, trace::DenseTrace};
use std::fs::File;
use std::io::BufReader;

#[macro_use]
extern crate maplit;

mod bfs_solver;

fn main() {
    let reader = BufReader::new(File::open("out/generator/trace.bin").expect("Opening trace.bin"));
    let trace_loaded = DenseTrace::read_bincode(reader).expect("Deserialize trace");

    let mut context =
        Context::load("./generator/assets/declarations.yaml").expect("Loading context");
    context.register_standard_operators();
    let rules = io::read_rules(&context, "./generator/assets/rules.txt", io::Mode::Normal);

    for calc in trace_loaded.unroll().take(1) {
        let initial = &calc.steps.last().unwrap().deduced;
        let end = &calc.steps.first().unwrap().initial;
        let solve_result = bfs_solver::solve(initial, end, &rules, 200);

        println!("Trying to solve {} => {}", initial, end);

        match solve_result.trace {
            Err(_) => println!("Could not solve {}", end),
            Ok(trace) => {
                for step in trace.iter() {
                    println!("{} using {}", step.term, step.rule);
                }
            }
        }

        println!(
            "\nNumber of fit calls: {}",
            solve_result.statistics.fit_calls_count
        );
        println!(
            "Number of fit tries: {}",
            solve_result.statistics.fits_count
        );
        println!(
            "Number of applies: {}",
            solve_result.statistics.applies_count
        );
    }
}
