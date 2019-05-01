use core::Symbol;
use std::collections::HashSet;
use std::collections::VecDeque;

fn list_all_nodes<'a>(symbol: &'a Symbol) -> Vec<&'a Symbol> {
    let mut queue: VecDeque<&Symbol> = VecDeque::new();
    queue.push_back(symbol);
    let mut result: Vec<&'a Symbol> = Vec::new();
    while !queue.is_empty() {
        let node = queue.pop_front().expect("At least one item");
        for child in node.childs.iter() {
            queue.push_back(child);
        }
        result.push(node);
    }
    result
}

// const alphabet: [&'static str; 26] = [
//     "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
//     "t", "u", "v", "w", "x", "y", "z",
// ];

/// In alphabetical order
fn next_variable(nodes: &[&Symbol]) -> Result<Symbol, &'static str> {
    let mut variables = HashSet::new();

    for variable in nodes.iter().filter(|node| node.childs.is_empty()) {
        if !variables.contains(&variable.ident) {
            variables.insert(&variable.ident);
        }
    }

    // Find first character in alphabet not contained in variables
    // TODO: Use macro for that
    let alphabet = (b'a'..b'z').map(|c| c as char).map(|c| c.to_string());
    for letter in alphabet {
        if !variables.contains(&letter) {
            return Ok(Symbol::new_variable(&letter, false));
        }
    }
    Err("No letter in alphabet found")
}

/// TODO: Consider using Memoization here
pub fn variables_generator<'a>(current: &'a Symbol) -> impl Fn() -> Vec<Symbol> + 'a {
    move || {
        let all_symbols = list_all_nodes(&current);
        let next = next_variable(&all_symbols).expect("");
        let mut symbols: Vec<Symbol> = all_symbols
            .iter()
            .filter(|s| s.ident != "=")
            .map(|s| (*s).clone())
            .collect();
        symbols.push(next);
        symbols
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use core::Context;
    #[test]
    fn list_all_nodes_simple() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c");

        let nodes = list_all_nodes(&symbol);

        assert_eq!(nodes.len(), 5);
        assert_eq!(
            nodes,
            vec![
                &symbol,
                &symbol.childs[0],
                &symbol.childs[1],
                &symbol.childs[0].childs[0],
                &symbol.childs[0].childs[1]
            ]
        );
    }

    #[test]
    fn next_variable_simple() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c");

        let nodes = list_all_nodes(&symbol);

        let actual = next_variable(&nodes).expect("Some variable");
        let expected = Symbol::parse(&context, "d");
        assert_eq!(actual, expected);
    }

    #[test]
    fn variables_generator_addition() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b");

        let actual = variables_generator(&symbol)();

        let expected = vec![
            Symbol::parse(&context, "a+b"),
            Symbol::parse(&context, "a"),
            Symbol::parse(&context, "b"),
            Symbol::parse(&context, "c"),
        ];

        assert_eq!(actual, expected);
    }

    #[test]
    fn variables_generator_omit_equation() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a=b");

        let actual = variables_generator(&symbol)();

        // Expect "=" being omitted
        let expected = vec![
            Symbol::parse(&context, "a"),
            Symbol::parse(&context, "b"),
            Symbol::parse(&context, "c"),
        ];

        assert_eq!(actual, expected);
    }
}
