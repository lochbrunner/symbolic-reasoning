use crate::common::merge_from;
use crate::fit::FitMap;
use crate::symbol::Symbol;
use std::collections::HashMap;
use std::slice::from_ref;

/// Maps all the symbols defined in the map with its new values
fn map_deep<'a, F>(
    mapping: &HashMap<&Symbol, &Symbol>,
    variable_creator: &F,
    orig: Symbol,
) -> Symbol
where
    F: Fn() -> &'a Symbol + Sized,
{
    match mapping.get(&orig) {
        None => {
            if orig.childs.is_empty() {
                if orig.fixed() {
                    orig
                } else {
                    // Introduce new variable
                    variable_creator().clone()
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

struct DeepMapFinding<'a> {
    symbol: Symbol,
    additional_mapping: HashMap<&'a Symbol, &'a Symbol>,
}

#[derive(Clone)]
struct ChildBranch<'a> {
    childs: Vec<Symbol>,
    current_mapping: HashMap<&'a Symbol, &'a Symbol>,
}

impl<'a> DeepMapFinding<'a> {
    pub fn no_mapping(symbol: Symbol) -> DeepMapFinding<'a> {
        DeepMapFinding {
            additional_mapping: HashMap::new(),
            symbol,
        }
    }
}

fn sub_hashmap<'a, 'b>(
    minuend: &'a mut HashMap<&Symbol, &Symbol>,
    subtrahend: &'b HashMap<&Symbol, &Symbol>,
) {
    for (key, _) in subtrahend.iter() {
        if minuend.contains_key(key) {
            minuend.remove(*key);
        }
    }
}

/// Maps all the symbols defined in the map with its new values
fn map_deep_batch<'a, F>(
    mapping: HashMap<&'a Symbol, &'a Symbol>,
    variable_creator: &F,
    orig: &'a Symbol,
) -> Vec<DeepMapFinding<'a>>
where
    F: Fn() -> Vec<&'a Symbol> + Sized,
{
    match mapping.get(&orig) {
        None => {
            if orig.childs.is_empty() {
                if orig.fixed() {
                    vec![DeepMapFinding::no_mapping(orig.clone())]
                } else {
                    // Introduce new variable
                    variable_creator()
                        .into_iter()
                        .map(|symbol| DeepMapFinding {
                            additional_mapping: hashmap! {orig => symbol},
                            symbol: symbol.clone(),
                        })
                        .collect()
                }
            } else {
                assert!(
                    orig.fixed(),
                    "Not fixed functions/operators not implemented yet!"
                );

                let mut scenarios = vec![ChildBranch {
                    childs: vec![],
                    current_mapping: mapping.clone(),
                }];

                for (i, child) in orig.childs.iter().enumerate() {
                    let mut additional_scenarios: Vec<ChildBranch> = Vec::new();
                    for scenario in scenarios.iter_mut() {
                        let mut patches = map_deep_batch(
                            scenario.current_mapping.clone(),
                            variable_creator,
                            child,
                        );
                        if patches.is_empty() {
                            unimplemented!();
                        } else {
                            let patch = patches.pop().expect("To be in");
                            scenario.childs.push(patch.symbol);
                            merge_from(&mut scenario.current_mapping, patch.additional_mapping);

                            // For the remaining scenarios create new branches
                            while !patches.is_empty() {
                                let patch = patches.pop().expect("To be in");

                                // Clone everything but the last child
                                let mut additional_scenario = ChildBranch {
                                    childs: [&scenario.childs[0..i], from_ref(&patch.symbol)]
                                        .concat(),
                                    current_mapping: scenario.current_mapping.clone(),
                                };
                                merge_from(
                                    &mut additional_scenario.current_mapping,
                                    patch.additional_mapping,
                                );
                                additional_scenarios.push(additional_scenario);
                            }
                        }
                    }
                    scenarios.append(&mut additional_scenarios);
                }
                // Convert ChildBranches to DeepMapFindings
                scenarios
                    .into_iter()
                    .map(|mut scenario| {
                        sub_hashmap(&mut scenario.current_mapping, &mapping);
                        DeepMapFinding {
                            additional_mapping: scenario.current_mapping,
                            symbol: Symbol {
                                depth: Symbol::calc_depth(&scenario.childs),
                                childs: scenario.childs,
                                flags: orig.flags,
                                value: orig.value,
                                ident: orig.ident.clone(),
                            },
                        }
                    })
                    .collect()
            }
        }
        Some(&value) => vec![DeepMapFinding::no_mapping(value.clone())],
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
pub fn apply<'a, F>(
    mapping: &FitMap,
    variable_creator: F,
    prev: &Symbol,
    conclusion: &Symbol,
) -> Symbol
where
    F: Fn() -> &'a Symbol + Sized,
{
    let FitMap { path, variable, .. } = mapping;
    // Adjust the conclusion
    let adjusted = map_deep(&variable, &variable_creator, conclusion.clone());
    deep_replace(path, prev, &adjusted)
}

pub fn apply_batch<'a, F>(
    mapping: &'a FitMap,
    variable_creator: F,
    prev: &Symbol,
    conclusion: &'a Symbol,
) -> Vec<Symbol>
where
    F: Fn() -> Vec<&'a Symbol> + Sized,
{
    let FitMap { path, variable, .. } = mapping;
    // Adjust the conclusion
    map_deep_batch(variable.clone(), &variable_creator, conclusion)
        .into_iter()
        .map(|dm| dm.symbol)
        .map(|adjusted| deep_replace(path, prev, &adjusted))
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
fn new_variable(ident: &str) -> Symbol {
    Symbol::new_variable(ident, false)
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
        let prev = Symbol::parse(&context, "A(a)").unwrap();
        let condition = Symbol::parse(&context, "A(b)").unwrap();
        let conclusion = Symbol::parse(&context, "B(b)").unwrap();

        let mapping = fit(&prev, &condition).pop().expect("One mapping");
        let var = new_variable("a");
        let actual = apply(&mapping, &|| &var, &prev, &conclusion);
        let expected = Symbol::parse(&context, "B(a)").unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn inner_hierarchically_variable() {
        let context = create_context(vec!["A", "B", "C"], vec![]);
        let prev = Symbol::parse(&context, "A(B(a))").unwrap();
        let condition = Symbol::parse(&context, "B(b)").unwrap();
        let conclusion = Symbol::parse(&context, "C(b)").unwrap();
        let expected = Symbol::parse(&context, "A(C(a))").unwrap();

        let mapping = fit(&prev, &condition).pop().expect("One mapping");
        let var = new_variable("a");
        let actual = apply(&mapping, &|| &var, &prev, &conclusion);
        assert_eq!(actual, expected);
    }

    #[test]
    fn function_hierarchical_function() {
        let context = create_context(vec!["A", "B", "C", "D"], vec!["d"]);
        let prev = Symbol::parse(&context, "A(B(C(a,b),C(a,b)))").unwrap();
        let condition = Symbol::parse(&context, "B(e,e)").unwrap();
        let conclusion = Symbol::parse(&context, "D(e)").unwrap();
        let expected = Symbol::parse(&context, "A(D(C(a,b)))").unwrap();

        let mapping = fit(&prev, &condition).pop().expect("One mapping");
        // Expect e -> C(a,b)
        let var = new_variable("a");
        let actual = apply(&mapping, &|| &var, &prev, &conclusion);
        assert_eq!(actual, expected);
    }

    #[test]
    fn readme_example() {
        let context = Context::standard();
        let initial = Symbol::parse(&context, "b*(c*d-c*d)=e").unwrap();
        let rule = Rule {
            condition: Symbol::parse(&context, "a-a").unwrap(),
            conclusion: Symbol::parse(&context, "0").unwrap(),
        };

        let mapping = fit(&initial, &rule.condition).pop().expect("One mapping");
        let var = new_variable("a");
        let actual = apply(&mapping, &|| &var, &initial, &rule.conclusion);
        let expected = Symbol::parse(&context, "b*0=e").unwrap();

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

    #[test]
    fn map_deep_simple() {
        let orig = new_variable("a");
        let a = new_variable("a");
        let b = new_variable("b");
        let mapping = hashmap! {
            &a => &b
        };
        let var = new_variable("a");

        let actual = map_deep(&mapping, &|| &var, orig);
        let expected = new_variable("b");

        assert_eq!(actual, expected);
    }

    #[test]
    fn deep_replace_simple() {
        let context = create_context(vec!["A", "B"], vec![]);
        let orig = Symbol::parse(&context, "A(b)").unwrap();
        let part = Symbol::parse(&context, "c").unwrap();
        let path = vec![0];
        let actual = deep_replace(&path, &orig, &part);

        let expected = Symbol::parse(&context, "A(c)").unwrap();

        println!("Actual:   {}", actual);
        println!("Expected: {}", expected);
        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_introduce_new_variable() {
        let context = Context::standard();
        let orig = Symbol::parse(&context, "a").unwrap();
        let conclusion = Symbol::parse(&context, "x").unwrap();

        let scenario = FitMap {
            path: vec![],
            variable: hashmap! {},
        };

        let var = new_variable("v");

        let actual = apply(&scenario, || &var, &orig, &conclusion);

        let expected = Symbol::parse(&context, "v").unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_batch_introduce_consistent_variable_simple() {
        let context = Context::standard();
        let parse = |formula: &str| -> Symbol { Symbol::parse(&context, formula).unwrap() };

        let orig = parse("a");
        let conclusion = parse("x");

        let scenario = FitMap {
            path: vec![],
            variable: hashmap! {},
        };

        let vars = vec![parse("v"), parse("w")];

        let actual = apply_batch(&scenario, || vars.iter().collect(), &orig, &conclusion);

        let expected = vec![parse("v"), parse("w")];

        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_batch_introduce_consistent_variable_consistent_flat() {
        // Issue #8
        let context = Context::standard();

        let parse = |formula: &str| -> Symbol { Symbol::parse(&context, formula).unwrap() };

        let orig = new_variable("a");
        let conclusion = parse("x-x");

        let scenario = FitMap {
            path: vec![],
            variable: hashmap! {},
        };

        let vars = vec![parse("v"), parse("w"), parse("u")];

        let actual = apply_batch(&scenario, || vars.iter().collect(), &orig, &conclusion);

        let expected = vec![parse("u-u"), parse("w-w"), parse("v-v")];

        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn apply_batch_introduce_consistent_variable_consistent_deep() {
        // Issue #8
        let context = Context::standard();

        let parse = |formula: &str| -> Symbol { Symbol::parse(&context, formula).unwrap() };

        let orig = new_variable("a");
        let conclusion = parse("x*x-x");

        let scenario = FitMap {
            path: vec![],
            variable: hashmap! {},
        };

        let vars = vec![parse("v"), parse("w"), parse("u")];

        let actual = apply_batch(&scenario, || vars.iter().collect(), &orig, &conclusion);

        let expected = vec![parse("v*v-v"), parse("w*w-w"), parse("u*u-u")];

        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }
}
