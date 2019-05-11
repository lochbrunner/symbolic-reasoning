use super::symbol::*;
use maplit::*;
use std::collections::HashMap;

/// Use this struct later
#[derive(Debug, Clone)]
pub struct FitMap<'a> {
    // TODO: Needs to be public?
    /// outer is key, inner is value
    pub variable: HashMap<&'a Symbol, &'a Symbol>,
    /// Where the inner root is located in the outer
    /// [Deprecated]
    pub location: &'a Symbol,
    /// Path to the node
    /// Each item represents the index of the next child
    pub path: Vec<usize>,
}

fn fit_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
    path: &[usize],
) -> Vec<FitMap<'a>> {
    if !inner.fixed() && !outer.only_root() {
        fit_abstract(outer, inner, map, path)
    } else {
        fit_fixed(outer, inner, map, path)
    }
}

/// Creates new FitMap without the previous mappings
fn fit_abstract<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
    path: &[usize],
) -> Vec<FitMap<'a>> {
    let contradiction = match map {
        Some(prev_map) => {
            prev_map.variable.contains_key(inner) && prev_map.variable[inner] != outer
        }
        None => false,
    };
    if contradiction || outer.only_root() {
        vec![]
    } else {
        // Add folking
        let fittings: Vec<FitMap<'a>> = vec![FitMap {
            variable: hashmap! {inner => outer},
            location: outer,
            path: path.to_vec(),
        }];
        folk_childs(outer, inner, map, path, fittings)
    }
}

/// Assuming for now target and source have length 1
#[allow(clippy::needless_range_loop, clippy::get_unwrap)]
fn add_extension<'a>(target: &mut Vec<FitMap<'a>>, source: Vec<FitMap<'a>>) {
    assert_eq!(
        target.len(),
        1,
        "Folking of extensions is not supported yet"
    );
    if source.is_empty() {
        target.clear();
    } else {
        'scenario: for i in 0..target.len() {
            let target_scenario = target.get_mut(i).unwrap();
            let source_scenario = &source[i];
            // Contradiction with previous?
            'conflicts: for (key, new_value) in source_scenario.variable.iter() {
                match target_scenario.variable.get(key) {
                    None => continue 'conflicts,
                    Some(old_value) => {
                        if new_value != old_value {
                            target.clear();
                            continue 'scenario;
                        }
                    }
                }
            }
            target_scenario
                .variable
                .extend(source_scenario.variable.iter());
        }
    }
}

/// Merges the mappings of one scenario
/// The usage of this function is not the most preferment approach
fn merge_mappings<'a>(prev: &FitMap<'a>, extension: &[FitMap<'a>]) -> Vec<FitMap<'a>> {
    let mut merged = Vec::new();

    for extension_scenario in extension.iter() {
        let mut target_scenario = FitMap {
            variable: HashMap::new(),
            location: prev.location,
            path: prev.path.clone(),
        };
        for (key, value) in extension_scenario.variable.iter() {
            target_scenario.variable.insert(*key, *value);
        }
        for (key, value) in prev.variable.iter() {
            target_scenario.variable.insert(*key, *value);
        }
        merged.push(target_scenario);
    }

    merged
}

fn folk_childs<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
    path: &[usize],
    mut fittings: Vec<FitMap<'a>>,
) -> Vec<FitMap<'a>> {
    // if (inner.only_root() || outer.only_root()) {
    //     return vec![];

    for (i, child) in outer.childs.iter().enumerate() {
        let mut path = path.to_vec();
        path.push(i);
        let branches = fit_impl(child, inner, map, &path);
        // Folk here
        fittings.extend(branches);
    }
    fittings
}

/// When folking give the child only the relevant branches
fn fit_fixed<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
    path: &[usize],
) -> Vec<FitMap<'a>> {
    // Check root
    if outer.ident != inner.ident {
        folk_childs(outer, inner, map, path, Vec::new())
    } else if outer.childs.len() != inner.childs.len() {
        // Wrong number of childs
        vec![]
    } else {
        // TODO: Dont allocate memory for hypothetical used variable
        let new_scenario = Some(FitMap {
            variable: HashMap::new(),
            location: outer,
            path: path.to_vec(),
        });
        let scenario = match map {
            Some(_) => map,
            None => &new_scenario,
        };

        // Operator matches
        // Check for variable repetition
        let mut extension: Vec<FitMap> = vec![FitMap {
            variable: HashMap::new(),
            location: outer, // or use from parent
            path: path.to_vec(),
        }];

        for i in 0..outer.childs.len() {
            let add = fit_impl(&outer.childs[i], &inner.childs[i], &scenario, path);
            if add.is_empty() {
                return vec![];
            }
            add_extension(&mut extension, add);
        }
        match scenario {
            Some(scenario) => merge_mappings(scenario, &extension),
            _ => vec![],
        }
    }
}

/// Finds scenarios in which the inner symbol fits into the outer
pub fn fit<'a>(outer: &'a Symbol, inner: &'a Symbol) -> Vec<FitMap<'a>> {
    let map = Option::<FitMap>::None;
    let path = vec![];
    fit_impl(outer, inner, &map, &path)
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::common::format_mapping;
    use crate::context::*;
    use test::Bencher;

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

    #[allow(dead_code)]
    fn format_scenario(fit: &FitMap) -> String {
        format_mapping(&fit.variable)
    }

    #[allow(dead_code)]
    fn format_scenarios(scenarios: &[FitMap]) -> String {
        scenarios
            .iter()
            .map(format_scenario)
            .map(|sc| format!("Scenario:\n {}", sc))
            .collect::<Vec<String>>()
            .join("\n")
    }

    fn new_variable(ident: &str) -> Symbol {
        Symbol::new_variable(ident, false)
    }

    #[test]
    fn symbol_fit_root_type() {
        let a = new_variable("a");

        let b = new_variable("b");

        let all_maps = fit(&a, &b); // Necessary to keep the vector in scope
        let map = all_maps.iter().nth(0).unwrap();
        assert_eq!(map.variable, hashmap!(&b => &a), "variable on variable");

        let a = Symbol {
            ident: String::from("a"),
            flags: 0,
            depth: 1,
            childs: Vec::new(),
            value: None,
        };

        let b = Symbol {
            ident: String::from("b"),
            flags: 0,
            depth: 1,
            childs: Vec::new(),
            value: None,
        };
        let oof = fit(&a, &b);
        assert_eq!(oof.len(), 1, "operator on operator");

        let a = new_variable("a");

        let b = Symbol {
            ident: String::from("b"),
            flags: 0,
            depth: 1,
            childs: Vec::new(),
            value: None,
        };

        let vof = fit(&a, &b);
        assert_eq!(vof.len(), 1, "variable on operator");

        let a = Symbol {
            ident: String::from("a"),
            flags: 0,
            depth: 1,
            childs: Vec::new(),
            value: None,
        };

        let b = new_variable("b");
        let ovf = fit(&a, &b);
        assert_eq!(ovf.len(), 1, "operator on variable");
    }

    #[test]
    fn operator_flat_single_variable() {
        // a => b
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(b)");
        let inner = Symbol::parse(&context, "A(a)");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        let FitMap {
            variable,
            location,
            path,
        } = mapping;

        assert_eq!(variable.len(), 1, "Expected one mapping");

        assert_eq!(location, &&outer);
        assert!(path.is_empty());
        assert_eq!(
            format!("{}", variable.keys().nth(0).unwrap()),
            "a",
            "wrong key"
        );
        assert_eq!(
            format!("{}", variable.values().nth(0).unwrap()),
            "b",
            "wrong value"
        );
    }

    #[test]
    fn operator_flat_multiple_variables_first() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a,b)");
        let inner = Symbol::parse(&context, "A(c,d)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        let scenario = scenarios.iter().nth(0).unwrap();

        assert_eq!(scenario.variable.len(), 2);

        assert_eq!(scenario.variable[&inner.childs[0]], &outer.childs[0]);
        assert_eq!(scenario.variable[&inner.childs[1]], &outer.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");
    }

    #[bench]
    fn inner_hierarchically_variable(b: &mut Bencher) {
        let context = create_context(vec!["A", "B"], vec![]);
        let outer = Symbol::parse(&context, "A(B(a))");
        let inner = Symbol::parse(&context, "B(b)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario.location, &outer.childs[0]);

        assert_eq!(scenario.variable.len(), 1);
        assert_eq!(
            scenario.variable[&inner.childs[0]],
            &outer.childs[0].childs[0]
        );
        assert_eq!(scenario.location, &outer.childs[0], "Wrong location");
        assert_eq!(scenario.path, vec![0], "Wrong path");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[bench]
    fn variable_maps_to_operator(b: &mut Bencher) {
        let context = create_context(vec!["A", "B", "C"], vec![]);
        let outer = Symbol::parse(&context, "A(B(C(b)))");
        let inner = Symbol::parse(&context, "B(a)");

        // Expect a -> C(b)

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1, "Expected one scenario");

        let scenario = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario.variable.len(), 1);
        let expected_key = Symbol::parse(&context, "a");
        assert!(
            scenario.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let expected_value = Symbol::parse(&context, "C(b)");
        let actual_value = scenario.variable.get(&expected_key).unwrap();
        assert_eq!(actual_value, &&expected_value, "Expect value to be C(b)");
        assert_eq!(format!("{}", actual_value), "C(b)");

        assert_eq!(scenario.location, &outer.childs[0], "Wrong location");
        assert_eq!(scenario.path, vec![0], "Wrong path");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[bench]
    fn complex_inner_simple(b: &mut Bencher) {
        let context = create_context(vec!["A", "B"], vec![]);
        let outer = Symbol::parse(&context, "A(B(a), b)");
        let inner = Symbol::parse(&context, "A(B(c), d)");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let scenario = all_mappings.iter().nth(0).unwrap();
        assert_eq!(scenario.variable.len(), 2, "Expected 2 mappings");

        let expected_key = Symbol::parse(&context, "c");
        assert!(
            scenario.variable.contains_key(&expected_key),
            "Expect mapping contains variable c"
        );
        let actual_value = scenario.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "a");

        let expected_key = Symbol::parse(&context, "d");
        assert!(
            scenario.variable.contains_key(&expected_key),
            "Expect mapping contains variable d"
        );
        let actual_value = scenario.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "b");
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[test]
    fn complex_inner_differ_variables() {
        let context = create_context(vec!["A", "B"], vec![]);
        let outer = Symbol::parse(&context, "A(B(b), c)");
        let inner = Symbol::parse(&context, "A(B(a), a)");

        assert!(
            fit(&outer, &inner).is_empty(),
            "Variables are contradictory"
        );
    }

    #[test]
    fn operator_flat_multiple_variables_same_target() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(c,c)");
        let inner = Symbol::parse(&context, "A(a,b)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1, "Exactly one scenario");
        let scenario = scenarios.iter().nth(0).unwrap();

        assert_eq!(scenario.variable.len(), 2);
        assert_eq!(scenario.variable[&inner.childs[0]], &outer.childs[0]);
        assert_eq!(scenario.variable[&inner.childs[1]], &outer.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");
    }

    #[bench]
    fn operator_flat_multiple_variables_same(b: &mut Bencher) {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a,a)");
        let inner = Symbol::parse(&context, "A(b,b)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        let scenario = scenarios.iter().nth(0).unwrap();

        assert_eq!(scenario.variable.len(), 1);

        assert_eq!(scenario.variable[&inner.childs[0]], &outer.childs[0]);
        assert_eq!(scenario.variable[&inner.childs[1]], &outer.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[test]
    fn operator_flat_multiple_variables_contradicting() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(c,d)");
        let inner = Symbol::parse(&context, "A(a,a)");

        assert!(fit(&outer, &inner).is_empty(), "Not injective");
    }

    #[test]
    fn operator_flat_wrong_childs() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a)");
        let inner = Symbol::parse(&context, "A(b,c)");

        assert!(fit(&outer, &inner).is_empty());
    }

    #[test]
    fn flat_operators() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a)");
        let inner = Symbol::parse(&context, "A(b)");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1, "Expected one scenario");

        let scenario = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario.variable.len(), 1, "Expected one variable mapping");
        assert_eq!(format_scenario(scenario), "b => a");
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");
    }

    #[test]
    fn flat_operators_different_arg_count() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a, b)");
        let inner = Symbol::parse(&context, "A(b)");

        assert!(fit(&outer, &inner).is_empty());
    }

    #[test]
    fn folk_simple() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a, b)");
        let inner = Symbol::parse(&context, "c");

        // Expect two scenarios (not order sensitive)
        // 1. c => A(a,b)
        // 2. c => a
        // 3. c => b

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 3);

        let scenario_root = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario_root.variable.len(), 1);
        assert!(scenario_root.variable.contains_key(&inner));
        assert_eq!(scenario_root.variable[&inner], &outer);
        assert_eq!(format_scenario(scenario_root), "c => A(a, b)");
        assert_eq!(scenario_root.location, &outer, "Wrong location");
        assert!(scenario_root.path.is_empty(), "Wrong path");

        let scenario_ac = scenarios.iter().nth(1).unwrap();
        assert_eq!(scenario_ac.variable.len(), 1);
        assert!(scenario_ac.variable.contains_key(&inner));
        assert_eq!(scenario_ac.variable[&inner], &outer.childs[0]);
        assert_eq!(format_scenario(scenario_ac), "c => a");
        assert_eq!(scenario_ac.location, &outer.childs[0], "Wrong location");
        assert_eq!(scenario_ac.path, vec![0], "Wrong path");

        let scenario_bc = scenarios.iter().nth(2).unwrap();
        assert_eq!(scenario_bc.variable.len(), 1);
        assert!(scenario_bc.variable.contains_key(&inner));
        assert_eq!(scenario_bc.variable[&inner], &outer.childs[1]);
        assert_eq!(format_scenario(scenario_bc), "c => b");
        assert_eq!(scenario_bc.location, &outer.childs[1], "Wrong location");
        assert_eq!(scenario_bc.path, vec![1], "Wrong path");
    }

    #[test]
    fn folk_zero() {
        let context = create_context(vec!["A"], vec!["B", "C", "c"]);
        let outer = Symbol::parse(&context, "A(B, C)");
        let inner = Symbol::parse(&context, "c");

        let scenarios = fit(&outer, &inner);
        // println!("Scenarios:\n{}", format_scenarios(&scenarios));
        assert!(scenarios.is_empty());
    }

    #[test]
    fn no_variable_simple() {
        let context = create_context(vec![], vec!["A"]);
        let outer = Symbol::parse(&context, "A");
        let inner = Symbol::parse(&context, "A");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        assert!(scenario.variable.is_empty());
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");
    }

    #[test]
    fn no_variable_flat() {
        let context = create_context(vec!["F"], vec!["A", "B", "C"]);
        let outer = Symbol::parse(&context, "F(A,B,C)");
        let inner = Symbol::parse(&context, "F(A,B,C)");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        assert!(scenario.variable.is_empty());
        assert_eq!(scenario.location, &outer, "Wrong location");
        assert!(scenario.path.is_empty(), "Wrong path");
    }

    #[bench]
    fn no_variable_deep(b: &mut Bencher) {
        let context = create_context(vec!["D", "F"], vec!["A", "B", "C"]);
        let outer = Symbol::parse(&context, "D(F(A,B,C))");
        let inner = Symbol::parse(&context, "F(A,B,C)");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        assert!(scenario.variable.is_empty());
        assert_eq!(scenario.location, &outer.childs[0], "Wrong location");
        assert_eq!(scenario.path, vec![0], "Wrong path");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[test]
    fn function_hierarchical_function() {
        // Expect e -> C(a,b)
        let context = create_context(vec!["A", "B", "C", "D"], vec!["d"]);
        let outer = Symbol::parse(&context, "A(B(C(a,b),C(a,b)))");
        let inner = Symbol::parse(&context, "B(e,e)");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario.variable.len(), 1);
        assert_eq!(format_scenario(scenario), "e => C(a, b)");
        assert_eq!(scenario.path, vec![0], "Wrong path");
    }

    #[test]
    fn respect_only_root() {
        let mut context = create_context(vec![], vec!["a", "b", "c"]);
        context.register_standard_operators();
        let outer = Symbol::parse(&context, "a=b");
        let inner = Symbol::parse(&context, "c");

        let scenarios = fit(&outer, &inner);
        println!("Scenarios:\n{}", format_scenarios(&scenarios));
        assert!(scenarios.is_empty(), "No match expected");
    }

    #[test]
    fn generator_issue_1() {
        let context = Context::standard();
        let outer = Symbol::parse(&context, "a=b");
        let inner = Symbol::parse(&context, "a");

        // Expect two scenarios:
        // 1. a=>a
        // 2. a=>b
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 2);

        let scenario_aa = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario_aa.variable.len(), 1);
        assert_eq!(format_scenario(scenario_aa), "a => a");
        assert_eq!(scenario_aa.path, vec![0], "Wrong path");

        let scenario_ab = scenarios.iter().nth(1).unwrap();
        assert_eq!(scenario_ab.variable.len(), 1);
        assert_eq!(format_scenario(scenario_ab), "a => b");
        assert_eq!(scenario_ab.path, vec![1], "Wrong path");
    }
}
