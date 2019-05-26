use super::{Context, Rule, Symbol};
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

pub enum Mode {
    Reversed,
    Normal,
}

pub fn read_rules(context: &Context, filename: &str, mode: Mode) -> Vec<Rule> {
    let file = File::open(filename).expect("Opening file of rules");
    let file = BufReader::new(&file);
    let mut rules = Vec::new();
    for (_, line) in file.lines().enumerate() {
        let line = line.expect("Line");
        let parts = line.split("//").collect::<Vec<&str>>();
        let code = parts[0];
        if !code.is_empty() {
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

                    rules.push(match mode {
                        Mode::Reversed => Rule {
                            conclusion: Symbol::parse(context, second),
                            condition: Symbol::parse(context, first),
                        },
                        Mode::Normal => Rule {
                            conclusion: Symbol::parse(context, first),
                            condition: Symbol::parse(context, second),
                        },
                    });
                }
            }
        }
    }
    rules
}

pub fn read_premises(context: &Context, filename: &str) -> Vec<Symbol> {
    let file = File::open(filename).expect("Opening file of premises");
    let file = BufReader::new(&file);
    let mut premises = Vec::new();
    for (_, line) in file.lines().enumerate() {
        let line = line.expect("Line");
        let parts = line.split("//").collect::<Vec<&str>>();
        let code = parts[0];
        if !code.is_empty() {
            premises.push(Symbol::parse(context, &code));
        }
    }
    premises
}
