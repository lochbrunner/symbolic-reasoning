use core::{Rule, Symbol};

fn initial_rules() -> Vec<Rule> {
    vec![
        Rule::new("A(a) => B(a,b)\0"),
        Rule::new("A(a,b) => B(a,b)\0"),
        Rule::new("A(a,b) => B(a,b)\0"),
    ]
}

fn init_prems() -> Vec<Symbol> {
    vec![Symbol::new("A(a)\0")]
}

fn main() {
    let rules = initial_rules();
    for rule in &rules {
        println!("Rule {}", rule);
    }

    let prems = init_prems();
    for prem in &prems {
        println!("Premise {}", prem);
    }
}
