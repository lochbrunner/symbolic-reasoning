use core::{Context, Rule, Symbol};

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

fn read_rules(context: &Context) -> Vec<Rule> {
    let file = File::open("./generator/assets/rules.txt").expect("Opening file of rules");
    let file = BufReader::new(&file);
    let mut rules = Vec::new();
    for (_, line) in file.lines().enumerate() {
        let line = line.expect("Line");
        let parts = line.split("//").collect::<Vec<&str>>();
        let code = parts[0];
        if code.len() > 0 {
            let mut parts = code.split("<=>").collect::<Vec<&str>>();
            if parts.len() == 2 {
                let first = parts.pop().expect("First part");
                let second = parts.pop().expect("Second part");

                rules.push(Rule {
                    conclusion: Symbol::parse(context, first),
                    condition: Symbol::parse(context, second),
                });

                rules.push(Rule {
                    conclusion: Symbol::parse(context, second),
                    condition: Symbol::parse(context, first),
                });
            } else {
                let mut parts = code.split("=>").collect::<Vec<&str>>();
                if parts.len() == 2 {
                    let first = parts.pop().expect("First part");
                    let second = parts.pop().expect("Second part");

                    rules.push(Rule {
                        conclusion: Symbol::parse(context, first),
                        condition: Symbol::parse(context, second),
                    });
                }
            }
        }
    }
    rules
}

fn read_premises(context: &Context) -> Vec<Symbol> {
    let file = File::open("./generator/assets/premises.txt").expect("Opening file of premises");
    let file = BufReader::new(&file);
    let mut premises = Vec::new();
    for (_, line) in file.lines().enumerate() {
        let line = line.expect("Line");
        let parts = line.split("//").collect::<Vec<&str>>();
        let code = parts[0];
        if code.len() > 0 {
            premises.push(Symbol::parse(context, &code));
        }
    }
    premises
}

fn main() {
    let context = Context::load("generator/assets/declarations.yaml");

    let rules = read_rules(&context);

    println!("Rules:");
    for rule in &rules {
        println!("  {}", rule);
    }

    let premises = read_premises(&context);
    println!("\nPremises:");

    for premise in &premises {
        println!("  {}", premise);
    }
}
