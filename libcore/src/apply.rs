use crate::context::{Context, Declaration};
use crate::fit::FitMap;
use crate::symbol::Symbol;
use std::collections::HashMap;

/// Maps all the symbols defined in the map with its new values
fn map_deep(mapping: &HashMap<&Symbol, &Symbol>, orig: Symbol) -> Symbol {
    match mapping.get(&orig) {
        None => {
            if orig.childs.is_empty() {
                orig
            } else {
                let childs = orig
                    .childs
                    .iter()
                    .map(|child| map_deep(mapping, child.clone()))
                    .collect::<Vec<Symbol>>();
                Symbol {
                    depth: Symbol::calc_depth(&childs),
                    childs,
                    fixed: orig.fixed,
                    value: orig.value,
                    ident: orig.ident.clone(),
                }
            }
        }
        Some(&value) => value.clone(),
    }
}

fn deep_replace_impl(path: &[usize], level: usize, orig: &Symbol, new: Symbol) -> Symbol {
    if level == path.len() {
        new.clone() // Only necessary because of map does not know that we only
    } else {
        let i = path[level];

        let mut childs = Vec::with_capacity(orig.childs.len());
        childs.extend_from_slice(&orig.childs[0..i]);
        childs.push(deep_replace_impl(path, level + 1, &orig.childs[i], new));
        childs.extend_from_slice(&orig.childs[i + 1..]);

        Symbol {
            depth: Symbol::calc_depth(&childs),
            childs,
            fixed: orig.fixed,
            value: orig.value,
            ident: orig.ident.clone(),
        }
    }
}

fn deep_replace(path: &[usize], orig: &Symbol, new: Symbol) -> Symbol {
    deep_replace_impl(path, 0, orig, new)
}

/// Applies the mapping on a expression in order generate a new expression
pub fn apply(mapping: &FitMap, prev: &Symbol, conclusion: &Symbol) -> Symbol {
    let FitMap { path, variable, .. } = mapping;
    // Adjust the conclusion
    let adjusted = map_deep(&variable, conclusion.clone());
    // println!("adjusted: {}", adjusted);

    deep_replace(path, prev, adjusted)
}

#[allow(dead_code)]
fn create_context(function_names: Vec<&str>, fixed_variable_names: Vec<&str>) -> Context {
    let mut declarations: HashMap<String, Declaration> = HashMap::new();
    for function_name in function_names.iter() {
        declarations.insert(
            String::from(*function_name),
            Declaration {
                is_fixed: true,
                is_function: true,
            },
        );
    }
    for fixed_variable_name in fixed_variable_names.iter() {
        declarations.insert(
            String::from(*fixed_variable_name),
            Declaration {
                is_fixed: true,
                is_function: false,
            },
        );
    }
    Context { declarations }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::fit::*;
    use crate::rule::Rule;

    #[allow(dead_code)]
    fn format_scenario(fit: &FitMap) -> String {
        fit.variable
            .iter()
            .map(|(source, target)| format!("{} => {}", source, target))
            .collect::<Vec<String>>()
            .join("\n")
    }

    #[test]
    fn operator_flat_single_variable() {
        let context = create_context(vec!["A", "B"], vec![]);
        let prev = Symbol::parse(&context, "A(a)");
        let condition = Symbol::parse(&context, "A(b)");
        let conclusion = Symbol::parse(&context, "B(b)");

        let mapping = fit(&prev, &condition).pop().expect("One mapping");

        let actual = apply(&mapping, &prev, &conclusion);
        let expected = Symbol::parse(&context, "B(a)");

        assert_eq!(actual, expected);
    }

    #[test]
    fn inner_hierarchically_variable() {
        let context = create_context(vec!["A", "B", "C"], vec![]);
        let prev = Symbol::parse(&context, "A(B(a))");
        let condition = Symbol::parse(&context, "B(b)");
        let conclusion = Symbol::parse(&context, "C(b)");
        let expected = Symbol::parse(&context, "A(C(a))");

        let mapping = fit(&prev, &condition).pop().expect("One mapping");

        let actual = apply(&mapping, &prev, &conclusion);
        assert_eq!(actual, expected);
    }

    #[test]
    fn function_hierarchical_function() {
        let context = create_context(vec!["A", "B", "C", "D"], vec!["d"]);
        let prev = Symbol::parse(&context, "A(B(C(a,b),C(a,b)))");
        let condition = Symbol::parse(&context, "B(e,e)");
        let conclusion = Symbol::parse(&context, "D(e)");
        let expected = Symbol::parse(&context, "A(D(C(a,b)))");

        let mapping = fit(&prev, &condition).pop().expect("One mapping");
        // Expect e -> C(a,b)
        let actual = apply(&mapping, &prev, &conclusion);
        assert_eq!(actual, expected);
    }

    #[test]
    fn readme_example() {
        let context = Context::standard();
        let initial = Symbol::parse(&context, "b*(c*d-c*d)=e");
        let rule = Rule {
            condition: Symbol::parse(&context, "a-a"),
            conclusion: Symbol::parse(&context, "0"),
        };

        let mapping = fit(&initial, &rule.condition).pop().expect("One mapping");

        let actual = apply(&mapping, &initial, &rule.conclusion);
        let expected = Symbol::parse(&context, "b*0=e");

        // println!("Mapping: {}", format_scenario(&mapping));
        // println!("Path: {:?}", mapping.path);
        // println!("Actual:   {}", actual);
        // println!("Expected: {}", expected);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod specs {
    use super::*;

    fn new_variable(ident: &str) -> Symbol {
        Symbol::new_variable(ident, false)
    }

    #[test]
    fn simple_map() {
        let orig = new_variable("a");
        let a = new_variable("a");
        let b = new_variable("b");
        let mapping = hashmap! {
            &a => &b
        };

        let actual = map_deep(&mapping, orig);
        let expected = new_variable("b");

        assert_eq!(actual, expected);
    }

    #[test]
    fn replace_simple() {
        let context = create_context(vec!["A", "B"], vec![]);
        let orig = Symbol::parse(&context, "A(b)");
        let part = Symbol::parse(&context, "c");
        let path = vec![0];
        let actual = deep_replace(&path, &orig, part);

        let expected = Symbol::parse(&context, "A(c)");

        println!("Actual:   {}", actual);
        println!("Expected: {}", expected);
        assert_eq!(actual, expected);
    }

    #[test]
    fn simple() {
        let symbol = new_variable("a");
        let _mapping = FitMap {
            variable: hashmap! {},
            location: &symbol,
            path: vec![],
        };
    }
}
