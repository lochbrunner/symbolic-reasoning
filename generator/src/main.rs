use core::{Context, Declaration, Rule, Symbol};

#[macro_use]
extern crate maplit;

fn initial_rules(context: &Context) -> Vec<Rule> {
    vec![
        Rule::parse(context, "A(a) => B(a,b)"),
        Rule::parse(context, "A(a,b) => B(a,b)"),
        Rule::parse(context, "A(a,b) => B(a,b)"),
    ]
}

fn init_prems(context: &Context) -> Vec<Symbol> {
    vec![Symbol::parse(context, "A(a)")]
}

fn main() {
    let context = Context {
        functions: hashmap! {
            String::from("A")=> Declaration{is_fixed:true, is_function: true},
            String::from("B")=> Declaration{is_fixed:true, is_function: true},
            String::from("C")=> Declaration{is_fixed:true, is_function: true},
            String::from("D")=> Declaration{is_fixed:true, is_function: true},
            String::from("E")=> Declaration{is_fixed:true, is_function: true},
            String::from("F")=> Declaration{is_fixed:true, is_function: true},
        },
    };

    let rules = initial_rules(&context);
    for rule in &rules {
        println!("Rule {}", rule);
    }

    let prems = init_prems(&context);
    for prem in &prems {
        println!("Premise {}", prem);
    }
}
