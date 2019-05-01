use crate::fit::FitMap;
use crate::symbol::Symbol;
use std::collections::HashMap;

/// Maps all the symbols defined in the map with its new values
fn map_deep(
    mapping: &HashMap<&Symbol, &Symbol>,
    variable_creator: fn() -> Symbol,
    orig: Symbol,
) -> Symbol {
    match orig.fixed() {
        _ => (),
    };

    match mapping.get(&orig) {
        None => {
            if orig.childs.is_empty() {
                if orig.fixed() {
                    orig
                } else {
                    // Introduce new variable
                    variable_creator()
                }
            } else {
                assert!(
                    orig.fixed(),
                    "Not fixed functions/operators not implemented yet!"
                );
                let childs = orig
                    .childs
                    .iter()
                    .map(|child| map_deep(mapping, variable_creator, child.clone()))
                    .collect::<Vec<Symbol>>();
                Symbol {
                    depth: Symbol::calc_depth(&childs),
                    childs,
                    flags: orig.flags,
                    value: orig.value,
                    ident: orig.ident.clone(),
                }
            }
        }
        Some(&value) => value.clone(),
    }
}

fn multiply_scenarios(factor: usize, prev: &mut Vec<Vec<Symbol>>) {
    assert!(factor > 0);
    if factor == 1 {
        return;
    }
    let orig: Vec<_> = prev.drain(..).collect();

    for item in orig.iter() {
        for _ in 0..factor {
            prev.push(item.clone());
        }
    }

    // *prev = prev
    //     .iter()
    //     .cycle()
    //     .take(factor * prev.len())
    //     .cloned()
    //     .collect();
}

fn push_child_scenarios_cycle(child: &[Symbol], scenarios: &mut Vec<Vec<Symbol>>) {
    for (scenario, child_scen) in scenarios.into_iter().zip(child.iter().cycle()) {
        scenario.push(child_scen.clone());
    }
}

/// Converts
/// from [childs][scenarios of different length]
/// to [scenarios equal length][childs]
fn fill_gaps_and_transpose(sparse: Vec<Vec<Symbol>>) -> Vec<Vec<Symbol>> {
    let mut scenarios: Vec<Vec<Symbol>> = vec![Vec::with_capacity(sparse.len())];
    // Find number of scenarios
    for (_i, child) in sparse.into_iter().enumerate() {
        multiply_scenarios(child.len(), &mut scenarios);
        push_child_scenarios_cycle(&child, &mut scenarios);
    }
    scenarios
}

/// Maps all the symbols defined in the map with its new values
fn map_deep_batch<F>(
    mapping: &HashMap<&Symbol, &Symbol>,
    variable_creator: &F,
    orig: Symbol,
) -> Vec<Symbol>
where
    F: Fn() -> Vec<Symbol> + Sized,
{
    match orig.fixed() {
        _ => (),
    };

    match mapping.get(&orig) {
        None => {
            if orig.childs.is_empty() {
                if orig.fixed() {
                    vec![orig]
                } else {
                    // Introduce new variable
                    variable_creator()
                }
            } else {
                assert!(
                    orig.fixed(),
                    "Not fixed functions/operators not implemented yet!"
                );
                // Folking here!
                let childs = orig
                    .childs
                    .iter()
                    .map(|child| map_deep_batch(mapping, variable_creator, child.clone()))
                    .collect::<Vec<Vec<Symbol>>>();

                fill_gaps_and_transpose(childs)
                    .into_iter()
                    .map(|childs| Symbol {
                        depth: Symbol::calc_depth(&childs),
                        childs,
                        flags: orig.flags,
                        value: orig.value,
                        ident: orig.ident.clone(),
                    })
                    .collect()
            }
        }
        Some(&value) => vec![value.clone()],
    }
}

fn deep_replace_impl(path: &[usize], level: usize, orig: &Symbol, new: &Symbol) -> Symbol {
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
            flags: orig.flags,
            value: orig.value,
            ident: orig.ident.clone(),
        }
    }
}

fn deep_replace(path: &[usize], orig: &Symbol, new: &Symbol) -> Symbol {
    deep_replace_impl(path, 0, orig, new)
}

/// Applies the mapping on a expression in order generate a new expression
/// * `variable_creator` - Is only used rarely that's why it should be evaluated lazy
/// * `prev` - The symbol which should be transformed to the new symbol
pub fn apply(
    mapping: &FitMap,
    variable_creator: fn() -> Symbol,
    prev: &Symbol,
    conclusion: &Symbol,
) -> Symbol {
    let FitMap { path, variable, .. } = mapping;
    // Adjust the conclusion
    let adjusted = map_deep(&variable, variable_creator, conclusion.clone());
    deep_replace(path, prev, &adjusted)
}

pub fn apply_batch<F>(
    mapping: &FitMap,
    variable_creator: F,
    prev: &Symbol,
    conclusion: &Symbol,
) -> Vec<Symbol>
where
    F: Fn() -> Vec<Symbol> + Sized,
{
    let FitMap { path, variable, .. } = mapping;
    // Adjust the conclusion
    let adjusteds = map_deep_batch(&variable, &variable_creator, conclusion.clone());
    adjusteds
        .iter()
        .map(|adjusted| deep_replace(path, prev, adjusted))
        .collect()
}

// Common code for testing

#[cfg(test)]
use crate::context::{Context, Declaration};

#[cfg(test)]
fn create_context(function_names: Vec<&str>, fixed_variable_names: Vec<&str>) -> Context {
    let mut declarations: HashMap<String, Declaration> = HashMap::new();
    for function_name in function_names.iter() {
        declarations.insert(
            String::from(*function_name),
            Declaration {
                is_fixed: true,
                is_function: true,
                only_root: false,
            },
        );
    }
    for fixed_variable_name in fixed_variable_names.iter() {
        declarations.insert(
            String::from(*fixed_variable_name),
            Declaration {
                is_fixed: true,
                is_function: false,
                only_root: false,
            },
        );
    }
    Context { declarations }
}
#[cfg(test)]
fn new_std_var() -> Symbol {
    Symbol::new_variable("a", false)
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

        let actual = apply(&mapping, new_std_var, &prev, &conclusion);
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

        let actual = apply(&mapping, new_std_var, &prev, &conclusion);
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
        let actual = apply(&mapping, new_std_var, &prev, &conclusion);
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

        let actual = apply(&mapping, new_std_var, &initial, &rule.conclusion);
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
    fn map_deep_simple() {
        let orig = new_variable("a");
        let a = new_variable("a");
        let b = new_variable("b");
        let mapping = hashmap! {
            &a => &b
        };

        let actual = map_deep(&mapping, new_std_var, orig);
        let expected = new_variable("b");

        assert_eq!(actual, expected);
    }

    #[test]
    fn deep_replace_simple() {
        let context = create_context(vec!["A", "B"], vec![]);
        let orig = Symbol::parse(&context, "A(b)");
        let part = Symbol::parse(&context, "c");
        let path = vec![0];
        let actual = deep_replace(&path, &orig, &part);

        let expected = Symbol::parse(&context, "A(c)");

        println!("Actual:   {}", actual);
        println!("Expected: {}", expected);
        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_introduce_new_variable() {
        let context = Context::standard();
        let orig = Symbol::parse(&context, "a");
        let conclusion = Symbol::parse(&context, "x");

        let scenario = FitMap {
            location: &orig,
            path: vec![],
            variable: hashmap! {},
        };

        let actual = apply(&scenario, || new_variable("v"), &orig, &conclusion);

        let expected = Symbol::parse(&context, "v");

        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_batch_introduce_consistent_variable_simple() {
        let context = Context::standard();
        let parse = |formula: &str| -> Symbol { Symbol::parse(&context, formula) };

        let orig = parse("a");
        let conclusion = parse("x");

        let scenario = FitMap {
            location: &orig,
            path: vec![],
            variable: hashmap! {},
        };

        let actual = apply_batch(
            &scenario,
            || vec![parse("v"), parse("w")],
            &orig,
            &conclusion,
        );

        let expected = vec![parse("v"), parse("w")];

        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_batch_introduce_consistent_variable_consistent() {
        let context = Context::standard();

        let parse = |formula: &str| -> Symbol { Symbol::parse(&context, formula) };

        let orig = new_variable("a");
        let conclusion = parse("x-x");

        let scenario = FitMap {
            location: &orig,
            path: vec![],
            variable: hashmap! {},
        };

        let actual = apply_batch(
            &scenario,
            || vec![new_variable("v"), new_variable("w")],
            &orig,
            &conclusion,
        );

        let expected = vec![parse("v-v"), parse("v-v")];

        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn fill_gaps_and_transpose_one_child_one_scenario() {
        let input = vec![vec![new_variable("a")]];
        let actual = fill_gaps_and_transpose(input);

        let expected = vec![vec![new_variable("a")]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn fill_gaps_and_transpose_one_child_multiple_scenario() {
        let input = vec![vec![new_variable("a"), new_variable("b")]];
        let actual = fill_gaps_and_transpose(input);

        let expected = vec![vec![new_variable("a")], vec![new_variable("b")]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn fill_gaps_and_transpose_multiple_child_one_scenario() {
        let input = vec![vec![new_variable("a")], vec![new_variable("b")]];
        let actual = fill_gaps_and_transpose(input);

        let expected = vec![vec![new_variable("a"), new_variable("b")]];
        assert_eq!(actual, expected);
    }

    #[test]
    fn fill_gaps_and_transpose_multiple_child_same_length_multiple_scenario() {
        let input = vec![
            vec![new_variable("c1_s1"), new_variable("c1_s2")],
            vec![new_variable("c2_s1"), new_variable("c2_s2")],
        ];
        let actual = fill_gaps_and_transpose(input);

        let expected = vec![
            vec![new_variable("c1_s1"), new_variable("c2_s1")],
            vec![new_variable("c1_s1"), new_variable("c2_s2")],
            vec![new_variable("c1_s2"), new_variable("c2_s1")],
            vec![new_variable("c1_s2"), new_variable("c2_s2")],
        ];
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn fill_gaps_and_transpose_multiple_child_diff_length_multiple_scenario() {
        let input = vec![
            vec![new_variable("c1_s1"), new_variable("c1_s2")],
            vec![
                new_variable("c2_s1"),
                new_variable("c2_s2"),
                new_variable("c2_s3"),
            ],
        ];

        let actual = fill_gaps_and_transpose(input);

        let expected = vec![
            vec![new_variable("c1_s1"), new_variable("c2_s1")],
            vec![new_variable("c1_s1"), new_variable("c2_s2")],
            vec![new_variable("c1_s1"), new_variable("c2_s3")],
            vec![new_variable("c1_s2"), new_variable("c2_s1")],
            vec![new_variable("c1_s2"), new_variable("c2_s2")],
            vec![new_variable("c1_s2"), new_variable("c2_s3")],
        ];
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }
}
