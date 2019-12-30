use core::bag::trace::DenseTrace;
use core::scenario::Scenario;
use std::fs::File;
use std::io::BufReader;

#[macro_use]
extern crate maplit;

mod bfs_solver;

fn main() {
    let reader = BufReader::new(
        File::open("out/generator/trace.bin").expect("Opening out/generator/trace.bin"),
    );
    let trace_loaded = DenseTrace::read_bincode(reader).expect("Deserialize trace");

    let scenario = Scenario::load_from_yaml("real_world_problems/basics/dataset.yaml").unwrap();
    let rules = scenario
        .rules
        .iter()
        .map(|(_, v)| v)
        .cloned()
        .collect::<Vec<_>>();

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
