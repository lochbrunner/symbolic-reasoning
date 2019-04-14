use core::{apply, fit, Context, Rule, Symbol};

use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

struct ApplyInfo<'a> {
    rule: &'a Rule,
    initial: &'a Symbol,
    deduced: Symbol,
}

impl<'a> ApplyInfo<'a> {
    fn print(&self) {
        let ded_str = format!("{}", self.deduced);
        let ini_str = format!("{}", self.initial);
        let rule = format!("{}", self.rule);
        println!("  {0: <10} | {1: <10} | {2: <10}", ded_str, ini_str, rule);
    }
}

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

fn deduce_once<'a>(initial: &'a Symbol, rule: &'a Rule) -> Vec<ApplyInfo<'a>> {
    fit(initial, &rule.condition)
        .iter()
        .map(|scenario| apply(scenario, initial, &rule.conclusion))
        .map(|deduced| ApplyInfo {
            rule,
            initial,
            deduced,
        })
        .collect()
}

fn deduce(initial: &Symbol, rules: &[Rule]) {
    let mut deduced: Vec<ApplyInfo> = Vec::new();

    for rule in rules.iter() {
        deduced.extend(deduce_once(initial, rule));
    }

    println!("Deduced:");
    for ded in deduced.iter() {
        ded.print();
    }
}

fn main() {
    let mut context = Context::load("generator/assets/declarations.yaml");
    context.register_standard_operators();

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

    let initial = &premises[0];

    deduce(initial, &rules);
}
