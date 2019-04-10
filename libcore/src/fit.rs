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
    pub location: &'a Symbol,
}

// How to indicate no fit? => Empty vector
fn fit_sym_op_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    maps: &Option<FitMap<'a>>,
) -> Vec<FitMap<'a>> {
    // TODO: Return iter later
    match outer.fixed {
        false => fit_var_sym_impl(outer, inner, maps),
        true => fit_op_op_impl(outer, inner, maps),
    }
}

fn fit_sym_sym_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
) -> Vec<FitMap<'a>> {
    match outer.fixed {
        true => fit_op_op_impl(outer, inner, map),
        false => fit_var_sym_impl(outer, inner, map),
    }
}

fn fit_var_sym_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
) -> Vec<FitMap<'a>> {
    // Do we still have a scenario?
    let mapping = match map {
        Some(prev_map) => {
            if prev_map.variable.contains_key(outer) {
                // println!("Map already contains {}", outer);
                FitMap {
                    variable: hashmap! {outer => inner},
                    location: outer, // TODO: Bad hack
                }
            } else {
                FitMap {
                    variable: hashmap! {outer => inner},
                    location: outer, // TODO: Bad hack
                }
            }
        }
        None => {
            FitMap {
                variable: hashmap! {outer => inner},
                location: outer, // TODO: Bad hack
            }
        }
    };
    vec![mapping]
}

pub fn fit<'a>(outer: &'a Symbol, inner: &'a Symbol) -> Vec<FitMap<'a>> {
    let map = Option::<FitMap>::None;
    fit_sym_sym_impl(outer, inner, &map)
}

/// Assuming for now target and source have length 1
fn add_extension<'a>(target: &mut Vec<FitMap<'a>>, source: Vec<FitMap<'a>>) -> () {
    assert_eq!(
        target.len(),
        1,
        "Folking of extensions is not supported yet"
    );
    if source.len() == 0 {
        target.clear();
    } else {
        'scenario: for i in 0..target.len() {
            let target_scenario = target.iter_mut().nth(i).unwrap();
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

/// The usage of this function is not the most preferment approach
fn merge_mappings<'a>(prev: &FitMap<'a>, extension: &Vec<FitMap<'a>>) -> Vec<FitMap<'a>> {
    let mut merged = Vec::new();

    for scenario in 0..extension.len() {
        let extension_scenario = &extension[scenario];
        let mut target_scenario = FitMap {
            variable: HashMap::new(),
            location: prev.location,
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

/// Does not folk yet
/// When folking give the child only the relevant branches
fn fit_op_op_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    map: &Option<FitMap<'a>>,
) -> Vec<FitMap<'a>> {
    // Check root
    if outer.ident != inner.ident {
        // Outer must be larger (For now)
        // if outer.depth > inner.depth {
        // Try fit the childs of the outer
        let mut fittings: Vec<FitMap<'a>> = Vec::new();
        for child in outer.childs.iter() {
            // let folk = FitMap {};
            let branches = fit_sym_op_impl(child, inner, map);
            // fittings.extend(merge_mappings(map, &branches));
            // Folk here
            fittings.extend(branches);
            //
        }
        fittings
    } else if outer.childs.len() != inner.childs.len() {
        // Wrong number of childs
        vec![]
    } else {
        // TODO: Dont allocate memory for hypothetical used variable
        let new_scenario = Some(FitMap {
            variable: HashMap::new(),
            location: outer,
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
        }];

        'childs: for i in 0..outer.childs.len() {
            let add = fit_sym_sym_impl(&outer.childs[i], &inner.childs[i], &scenario);
            add_extension(&mut extension, add);
        }
        match scenario {
            Some(scenario) => merge_mappings(scenario, &extension),
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::context::*;
    use test::Bencher;

    fn create_context(function_names: Vec<&str>, fixed_variable_names: Vec<&str>) -> Context {
        let mut functions: HashMap<String, Declaration> = HashMap::new();
        for function_name in function_names.iter() {
            functions.insert(
                String::from(*function_name),
                Declaration {
                    is_fixed: true,
                    is_function: true,
                },
            );
        }
        for fixed_variable_name in fixed_variable_names.iter() {
            functions.insert(
                String::from(*fixed_variable_name),
                Declaration {
                    is_fixed: true,
                    is_function: false,
                },
            );
        }
        Context { functions }
    }

    fn format_scenario(fit: &FitMap) -> String {
        fit.variable
            .iter()
            .map(|(source, target)| format!("{} => {}", source, target))
            .collect::<Vec<String>>()
            .join("\n")
    }

    // fn format_scenarios(scenarios: &Vec<FitMap>) -> String {
    //     scenarios
    //         .iter()
    //         .map(format_scenario)
    //         .map(|sc| format!("Scenario:\n {}", sc))
    //         .collect::<Vec<String>>()
    //         .join("\n")
    // }

    #[test]
    fn symbol_fit_root_type() {
        let a = Symbol::new_variable("a");

        let b = Symbol::new_variable("a");

        let all_maps = fit(&a, &b); // Necessary to keep the vector in scope
        let map = all_maps.iter().nth(0).unwrap();
        assert_eq!(map.variable, hashmap!(&a => &b), "variable on variable");

        let a = Symbol {
            ident: String::from("a"),
            fixed: true,
            depth: 1,
            childs: Vec::new(),
        };

        let b = Symbol {
            ident: String::from("a"),
            fixed: true,
            depth: 1,
            childs: Vec::new(),
        };
        let oof = fit(&a, &b);
        assert_eq!(oof.len(), 1, "operator on operator");

        let a = Symbol::new_variable("a");

        let b = Symbol {
            ident: String::from("a"),
            fixed: true,
            depth: 1,
            childs: Vec::new(),
        };

        let vof = fit(&a, &b);
        assert_eq!(vof.len(), 1, "variable on operator");

        let a = Symbol {
            ident: String::from("a"),
            fixed: true,
            depth: 1,
            childs: Vec::new(),
        };

        let b = Symbol::new_variable("a");
        let ovf = fit(&a, &b);
        assert_eq!(ovf.len(), 1, "operator on variable");
    }

    #[test]
    fn operator_flat_single_variable() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a)");
        let inner = Symbol::parse(&context, "A(b)");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        let FitMap { variable, location } = mapping;

        assert_eq!(variable.len(), 1, "Expected one mapping");

        assert_eq!(location, &&outer);
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
    fn operator_flat_multiple_variables() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a,b)");
        let inner = Symbol::parse(&context, "A(c,d)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        let scenario = scenarios.iter().nth(0).unwrap();

        assert_eq!(scenario.variable.len(), 2);

        assert_eq!(scenario.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(scenario.variable[&outer.childs[1]], &inner.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");
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

        // Expect a => b
        assert_eq!(scenario.variable.len(), 1);
        assert_eq!(
            scenario.variable[&outer.childs[0].childs[0]],
            &inner.childs[0]
        );
        assert_eq!(scenario.location, &outer.childs[0], "Wrong location");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[bench]
    fn variable_maps_to_operator(b: &mut Bencher) {
        let context = create_context(vec!["A", "B", "C"], vec![]);
        let outer = Symbol::parse(&context, "A(B(a))");
        let inner = Symbol::parse(&context, "B(C(b))");

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

        let expected_key = Symbol::parse(&context, "a");
        assert!(
            scenario.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let actual_value = scenario.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "c");

        let expected_key = Symbol::parse(&context, "b");
        assert!(
            scenario.variable.contains_key(&expected_key),
            "Expect mapping contains variable b"
        );
        let actual_value = scenario.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "d");
        assert_eq!(scenario.location, &outer, "Wrong location");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[test]
    fn complex_inner_differ_variables() {
        let context = create_context(vec!["A", "B"], vec![]);
        let outer = Symbol::parse(&context, "A(B(a), a)");
        let inner = Symbol::parse(&context, "A(B(b), c)");

        assert!(
            fit(&outer, &inner).is_empty(),
            "Variables are contradictory"
        );
    }

    #[test]
    fn operator_flat_multiple_variables_same_target() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a,b)");
        let inner = Symbol::parse(&context, "A(c,c)");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        let scenario = scenarios.iter().nth(0).unwrap();

        assert_eq!(scenario.variable.len(), 2);
        assert_eq!(scenario.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(scenario.variable[&outer.childs[1]], &inner.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");
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

        assert_eq!(scenario.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(scenario.variable[&outer.childs[1]], &inner.childs[1]);
        assert_eq!(scenario.location, &outer, "Wrong location");

        b.iter(|| {
            fit(&outer, &inner);
        })
    }

    #[test]
    fn operator_flat_multiple_variables_contradicting() {
        let context = create_context(vec!["A"], vec![]);
        let outer = Symbol::parse(&context, "A(a,a)");
        let inner = Symbol::parse(&context, "A(c,d)");

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
    fn operator_flat_inner_too_large() {
        let context = create_context(vec!["A", "B", "C"], vec![]);
        let outer = Symbol::parse(&context, "A(a)");
        let inner = Symbol::parse(&context, "B(C(b))");

        let scenarios = fit(&outer, &inner);
        // Expect a => "B(C(b))
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();
        let FitMap { variable, .. } = scenario;
        assert_eq!(variable.len(), 1);;

        assert!(
            variable.contains_key(&Symbol::parse(&context, "a")),
            "Expect mapping contains variable a"
        );

        assert_eq!(
            variable.get(&Symbol::parse(&context, "a")).unwrap(),
            &&Symbol::parse(&context, "B(C(b))")
        );
        assert_eq!(scenario.location, &outer.childs[0], "Wrong location");
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
        assert_eq!(format_scenario(scenario), "a => b");
        assert_eq!(scenario.location, &outer, "Wrong location");
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
        // 1. a => c
        // 2. b => c

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 2);

        let scenario_ac = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario_ac.variable.len(), 1);
        assert_eq!(scenario_ac.variable[&outer.childs[0]], &inner);
        assert_eq!(format_scenario(scenario_ac), "a => c");
        assert_eq!(scenario_ac.location, &outer.childs[0], "Wrong location");

        let scenario_bc = scenarios.iter().nth(1).unwrap();
        assert_eq!(scenario_bc.variable.len(), 1);
        assert_eq!(scenario_bc.variable[&outer.childs[1]], &inner);
        assert_eq!(format_scenario(scenario_bc), "b => c");
        assert_eq!(scenario_bc.location, &outer.childs[1], "Wrong location");
    }

    #[test]
    fn folk_inner_fixed() {
        let context = create_context(vec!["A"], vec!["C"]);
        let outer = Symbol::parse(&context, "A(a, b)");
        let inner = Symbol::parse(&context, "C");

        // Expect two scenarios (not order sensitive)
        // 1. a => C
        // 2. b => C

        let scenarios = fit(&outer, &inner);

        assert_eq!(scenarios.len(), 2);

        let scenario_ac = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario_ac.variable.len(), 1);
        assert_eq!(scenario_ac.variable[&outer.childs[0]], &inner);
        assert_eq!(format_scenario(scenario_ac), "a => C");
        assert_eq!(scenario_ac.location, &outer.childs[0], "Wrong location");

        let scenario_bc = scenarios.iter().nth(1).unwrap();
        assert_eq!(scenario_bc.variable.len(), 1);
        assert_eq!(scenario_bc.variable[&outer.childs[1]], &inner);
        assert_eq!(format_scenario(scenario_bc), "b => C");
        assert_eq!(scenario_bc.location, &outer.childs[1], "Wrong location");
    }

    #[test]
    fn folk_zero() {
        let context = create_context(vec!["A"], vec!["B", "C"]);
        let outer = Symbol::parse(&context, "A(B, C)");
        let inner = Symbol::parse(&context, "c");

        let scenarios = fit(&outer, &inner);
        // println!("Scenarios:\n{}", format_scenarios(&scenarios));
        assert!(scenarios.is_empty());
    }

    #[test]
    fn folk_some_inner_fixed() {
        let context = create_context(vec!["A"], vec!["B"]);
        let outer = Symbol::parse(&context, "A(B, c)");
        let inner = Symbol::parse(&context, "d");

        // Expect two scenarios (not order sensitive)
        // 1. a => C
        // 2. b => C

        let scenarios = fit(&outer, &inner);

        assert_eq!(scenarios.len(), 1);

        let scenario = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario.variable.len(), 1);
        assert_eq!(scenario.variable[&outer.childs[1]], &inner);
        assert_eq!(format_scenario(scenario), "c => d");
        assert_eq!(scenario.location, &outer.childs[1], "Wrong location");
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

        b.iter(|| {
            fit(&outer, &inner);
        })
    }
}
