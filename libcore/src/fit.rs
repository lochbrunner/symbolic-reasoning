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
fn fit_sym_op_impl<'a>(outer: &'a Symbol, inner: &'a Symbol, maps: &FitMap<'a>) -> Vec<FitMap<'a>> {
    // TODO: Return iter later
    match outer.fixed {
        false => fit_var_sym_impl(outer, inner, maps),
        true => fit_op_op_impl(outer, inner, maps),
    }
}

fn fit_sym_sym_impl<'a>(outer: &'a Symbol, inner: &'a Symbol, map: &FitMap<'a>) -> Vec<FitMap<'a>> {
    match outer.fixed {
        true => fit_op_op_impl(outer, inner, map),
        false => fit_var_sym_impl(outer, inner, map),
    }
}

fn fit_var_sym_impl<'a>(outer: &'a Symbol, inner: &'a Symbol, map: &FitMap<'a>) -> Vec<FitMap<'a>> {
    let mapping = if map.variable.contains_key(outer) {
        FitMap {
            variable: hashmap! {outer => inner},
            location: outer, // TODO: Bad hack
        }
    } else {
        FitMap {
            variable: hashmap! {outer => inner},
            location: outer, // TODO: Bad hack
        }
    };
    vec![mapping]
}

pub fn fit<'a>(outer: &'a Symbol, inner: &'a Symbol) -> Vec<FitMap<'a>> {
    let map = FitMap {
        variable: HashMap::new(),
        location: outer, // TODO: Bad hack
    };

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
fn fit_op_op_impl<'a>(outer: &'a Symbol, inner: &'a Symbol, map: &FitMap<'a>) -> Vec<FitMap<'a>> {
    // Check root
    if outer.ident != inner.ident {
        // Outer must be larger (For now)
        // if outer.depth > inner.depth {
        // Try fit the childs of the outer
        let mut fittings: Vec<FitMap<'a>> = Vec::new();
        for child in outer.childs.iter() {
            // TODO: Folk here?
            let branches = fit_sym_op_impl(child, inner, map);
            fittings.extend(merge_mappings(map, &branches));
        }
        fittings
    } else if outer.childs.len() != inner.childs.len() {
        // Wrong number of childs
        // Is it expected that this could happen?
        vec![]
    } else {
        // Operator matches
        // Check for variable repetition
        let mut extension: Vec<FitMap> = vec![FitMap {
            variable: HashMap::new(),
            location: outer, // or use from parent
        }];

        'childs: for i in 0..outer.childs.len() {
            // TODO: What when they folk?
            // Ignore folks for now
            let add = fit_sym_sym_impl(&outer.childs[i], &inner.childs[i], map);
            add_extension(&mut extension, add);
        }
        merge_mappings(&map, &extension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_scenario(fit: &FitMap) -> String {
        fit.variable
            .iter()
            .map(|(source, target)| format!("{} => {}", source, target))
            .collect::<Vec<String>>()
            .join("\n")
    }

    fn format_scenarios(scenarios: &Vec<FitMap>) -> String {
        scenarios
            .iter()
            .map(format_scenario)
            .map(|sc| format!("Scenario:\n {}", sc))
            .collect::<Vec<String>>()
            .join("\n")
    }

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
        let outer = Symbol::parse("A(a)\0");
        let inner = Symbol::parse("A(b)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        let FitMap { variable, .. } = mapping;

        assert_eq!(variable.len(), 1, "Expected one mapping");

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
        let outer = Symbol::parse("A(a,b)\0");
        let inner = Symbol::parse("A(c,d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 2);

        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn inner_hierarchically_variable() {
        let outer = Symbol::parse("A(B(a))\0");
        let inner = Symbol::parse("B(b)\0");

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let scenario = scenarios.iter().nth(0).unwrap();

        // Expect a => b
        assert_eq!(scenario.variable.len(), 1);
        assert_eq!(
            scenario.variable[&outer.childs[0].childs[0]],
            &inner.childs[0]
        );
    }

    #[test]
    fn variable_maps_to_operator() {
        let outer = Symbol::parse("A(B(a))\0");
        let inner = Symbol::parse("B(C(b))\0");

        // Expect a -> C(b)

        // Necessary to keep the vector in scope
        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1, "Expected one scenario");

        let mapping = scenarios.iter().nth(0).unwrap();
        assert_eq!(mapping.variable.len(), 1);
        let expected_key = Symbol::parse("a\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let expected_value = Symbol::parse("C(b)\0");
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(actual_value, &&expected_value, "Expect value to be C(b)");
        assert_eq!(format!("{}", actual_value), "C(b)");
    }

    #[test]
    fn complex_inner_simple() {
        let outer = Symbol::parse("A(B(a), b)\0");
        let inner = Symbol::parse("A(B(c), d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        assert_eq!(mapping.variable.len(), 2, "Expected 2 mappings");

        let expected_key = Symbol::parse("a\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "c");

        let expected_key = Symbol::parse("b\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable b"
        );
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "d");
    }

    #[test]
    fn complex_inner_differ_variables() {
        let outer = Symbol::parse("A(B(a), a)\0");
        let inner = Symbol::parse("A(B(b), c)\0");

        assert!(
            fit(&outer, &inner).is_empty(),
            "Variables are contradictory"
        );
    }

    #[test]
    fn operator_flat_multiple_variables_same_target() {
        let outer = Symbol::parse("A(a,b)\0");
        let inner = Symbol::parse("A(c,c)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 2);
        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn operator_flat_multiple_variables_same() {
        let outer = Symbol::parse("A(a,a)\0");
        let inner = Symbol::parse("A(b,b)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 1);

        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn operator_flat_multiple_variables_contradicting() {
        let outer = Symbol::parse("A(a,a)\0");
        let inner = Symbol::parse("A(c,d)\0");

        assert!(fit(&outer, &inner).is_empty(), "Not injective");
    }

    #[test]
    fn operator_flat_wrong_childs() {
        let outer = Symbol::parse("A(a)\0");
        let inner = Symbol::parse("A(b,c)\0");

        assert!(fit(&outer, &inner).is_empty());
    }

    #[test]
    fn operator_flat_inner_too_large() {
        let outer = Symbol::parse("A(a)\0");
        let inner = Symbol::parse("B(C(b))\0");

        let scenarios = fit(&outer, &inner);
        assert_eq!(scenarios.len(), 1);
        let mapping = scenarios.iter().nth(0).unwrap();
        let FitMap { variable, .. } = mapping;
        assert_eq!(variable.len(), 1);;

        assert!(
            variable.contains_key(&Symbol::parse("a\0")),
            "Expect mapping contains variable a"
        );

        assert_eq!(
            variable.get(&Symbol::parse("a\0")).unwrap(),
            &&Symbol::parse("B(C(b))\0")
        );
    }

    #[test]
    fn flat_operators() {
        let outer = Symbol::parse("A(a)\0");
        let inner = Symbol::parse("A(b)\0");

        assert_eq!(fit(&outer, &inner).len(), 1);
    }

    #[test]
    fn flat_operators_different_arg_count() {
        let outer = Symbol::parse("A(a, b)\0");
        let inner = Symbol::parse("A(b)\0");

        assert!(fit(&outer, &inner).is_empty());
    }

    #[test]
    fn folk() {
        let outer = Symbol::parse("A(a, b)\0");
        let inner = Symbol::parse("c\0");

        // Expect two scenarios (not order sensitive)
        // 1. a => c
        // 2. b => c

        let scenarios = fit(&outer, &inner);
        println!("Scenarios:\n{}", format_scenarios(&scenarios));

        assert_eq!(scenarios.len(), 2);

        let scenario_ac = scenarios.iter().nth(0).unwrap();
        assert_eq!(scenario_ac.variable.len(), 1);
        assert_eq!(scenario_ac.variable[&outer.childs[0]], &inner);
        assert_eq!(format_scenario(scenario_ac), "a => c");

        let scenario_bc = scenarios.iter().nth(1).unwrap();
        assert_eq!(scenario_bc.variable.len(), 1);
        assert_eq!(scenario_bc.variable[&outer.childs[1]], &inner);
        assert_eq!(format_scenario(scenario_bc), "b => c");
    }
}
