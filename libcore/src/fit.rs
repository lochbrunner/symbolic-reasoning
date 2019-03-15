use super::symbol::*;
// use std::fmt;
// use std::hash::Hash;

macro_rules! map(
    { $($key:expr => $value:expr),+ } => {
        {
            let mut m = ::std::collections::HashMap::new();
            $(
                m.insert($key, $value);
            )+
            m
        }
     };
);

use maplit::*;
use std::collections::HashMap;

pub fn format_map(fit: &FitMap) -> String {
    // Needed because of bug:
    // note: type must be known at this point
    let strings: Vec<String> = fit
        .variable
        .iter()
        .map(|(source, target)| format!("{} => {}", source, target))
        .collect();

    strings.join("\n")
}

/// Use this struct later
#[derive(Debug, Clone)]
pub struct FitMap<'a> {
    // TODO: Needs to be public?
    /// outer is key
    pub variable: HashMap<&'a Symbol, &'a Symbol>,
    // location: &'a Symbol,
}
/// Example i fits in O(o) with o => i
fn fit_op_var_impl<'a>(
    _outer: &'a Symbol,
    _inner: &'a Symbol,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    // For no ignore folks would be needed
    maps.iter().map(|_| Err("")).collect()
}

// How to indicate no fit? => Empty vector
fn fit_sym_op_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Operator,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    // TODO: Return iter later
    match outer {
        Symbol::Variable(_) => maps.iter().map(|_| Err("")).collect(),
        Symbol::Operator(outer_op) => fit_op_op_impl(outer_op, inner, maps),
    }
}

fn fit_sym_var_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    match outer {
        Symbol::Variable(_) => fit_var_var_impl(outer, inner, maps),
        Symbol::Operator(_) => fit_op_var_impl(outer, inner, maps),
    }
}

fn fit_sym_sym_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    match inner {
        Symbol::Operator(inner) => fit_sym_op_impl(outer, inner, maps),
        Symbol::Variable(_) => fit_sym_var_impl(outer, inner, maps),
    }
    // vec![]
}

fn fit_var_var_impl<'a>(
    outer: &'a Symbol,
    inner: &'a Symbol,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    // println!("outer var: {} inner var: {}", outer, inner);
    maps.iter()
        .map(|map| {
            // Contradiction to previous?
            // let outer_symbol = &{Symbol::Variable(*outer.clone())};
            if map.variable.contains_key(outer) {
                Ok(FitMap {
                    variable: map! {outer => inner},
                })
            } else {
                Ok(FitMap {
                    variable: map! {outer => inner},
                })
            }
            // Ok(FitMap {
            //     variable: HashMap::new(),
            // })
        })
        .collect()

    // if (outer == inner) {
    //     maps.iter().map(|| Ok({ FitMap }))
    // }
    // vec![]
}

fn fit_operator<'a>(outer: &'a Operator, inner: &'a Operator) -> Vec<FitMap<'a>> {
    // let mut mapping = HashMap::new();
    let map = vec![FitMap {
        variable: HashMap::new(),
    }];

    // Not very performant
    let mut result = Vec::new();
    for scenario in fit_op_op_impl(outer, inner, &map).iter() {
        match scenario {
            Ok(scenario) => result.push(scenario.clone()),
            _ => {}
        }
    }

    result
}

// fn merge_hashmaps<K: Hash + Eq + Clone, V: Clone>(
//     target: &mut HashMap<K, V>,
//     extension: &HashMap<K, V>,
// ) {
//     target.extend(extension.into_iter().map(|(k, v)| (k.clone(), v.clone())));
// }

fn add_extension<'a>(
    target: &mut Vec<Result<FitMap<'a>, &'static str>>,
    source: Vec<Result<FitMap<'a>, &'static str>>,
) -> () {
    for i in 0..target.len() {
        let target_scenario = target.iter_mut().nth(i).unwrap();
        let source_scenario = &source[i];

        if let Ok(target_scenario) = target_scenario {
            if let Ok(source_scenario) = source_scenario {
                target_scenario
                    .variable
                    .extend(source_scenario.variable.iter());
            } else {
                target[i] = Err("");
            }
        }
    }
}

/// The usage of this function is not the most preferment approach
fn merge_mappings<'a>(
    prev: &Vec<FitMap<'a>>,
    extension: &Vec<Result<FitMap<'a>, &'static str>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    assert_eq!(
        prev.len(),
        extension.len(),
        "Scenario vectors should have same length"
    );

    let mut merged = Vec::new();

    for scenario in 0..extension.len() {
        let extension_scenario = &extension[scenario];
        let prev_scenario = &prev[scenario];
        match extension_scenario {
            Ok(extension_scenario) => {
                let mut target_scenario = FitMap {
                    variable: HashMap::new(),
                };
                for (key, value) in extension_scenario.variable.iter() {
                    target_scenario.variable.insert(*key, *value);
                    println!("Adding from extension: {} => {}", key, value);
                }
                for (key, value) in prev_scenario.variable.iter() {
                    target_scenario.variable.insert(*key, *value);
                    println!("Adding from prev: {} => {}", key, value);
                }
                merged.push(Ok(target_scenario));
            }
            _ => merged.push(Err("Extension got had an error")),
        }
    }
    merged
}

/// Does not folk yet
/// When folking give the child only the relevant branches
fn fit_op_op_impl<'a>(
    outer: &'a Operator,
    inner: &'a Operator,
    maps: &Vec<FitMap<'a>>,
) -> Vec<Result<FitMap<'a>, &'static str>> {
    // let mut mapping = HashMap::new();
    // Check root
    if outer.ident != inner.ident {
        // Outer must be larger
        // if outer.depth > inner.depth {
        // Try fit the childs of the outer
        for child in outer.childs.iter() {
            // TODO: Folk here?
            // let mut folk = maps.clone();
            for branch in fit_sym_op_impl(child, inner, maps).iter() {
                match branch {
                    Ok(_b) => (),
                    Err(_) => (),
                }
            }
        }
        vec![]
    } else if outer.childs.len() != inner.childs.len() {
        // Wrong number of childs
        // Is it expected that this could happen?
        //maps.clear(); Return array with Err of same length than map
        vec![]
    } else {
        // Operator matches
        // Check for variable repetition
        let mut extension: Vec<Result<FitMap, &str>> = maps
            .iter()
            .map(|_| {
                Ok(FitMap {
                    variable: HashMap::new(),
                })
            })
            .collect();

        'childs: for i in 0..outer.childs.len() {
            println!("Childs: {} -> {}", outer.childs[i], inner.childs[i]);
            // TODO: do not override prev extension
            let add = fit_sym_sym_impl(&outer.childs[i], &inner.childs[i], maps);
            // extension.append(add);
            add_extension(&mut extension, add);
            // match &outer.childs[i] {
            // Symbol::Operator(oo) => {
            //     // Check inner operator
            //     print!("Outer: {} inner: {}\n", outer.childs[i], inner.childs[i]);
            //     if outer.childs[i] == inner.childs[i] {
            //         if let Symbol::Operator(io) = &inner.childs[i] {
            //             fit_operator_impl(oo, io, maps)
            //         } else {
            //             break 'childs;
            //         }
            //     } else {
            //         break 'childs;
            //     }
            // }
            // Symbol::Variable(_) => {
            //     // Check for injective
            //     for scenario in extension.iter_mut() {
            //         match scenario {
            //             Ok(scenario) => {
            //                 if scenario.variable.contains_key(&outer.childs[i]) {
            //                     // Do we still have this mapping?
            //                     if maps[0].variable.get(&outer.childs[i]).unwrap()
            //                         == &&inner.childs[i]
            //                     {
            //                         // continue;
            //                         continue 'childs;
            //                     } else {
            //                         break 'childs;
            //                     }
            //                 }
            //                 // Check for surjective
            //                 scenario.variable.insert(&outer.childs[i], &inner.childs[i]);
            //             }
            //             _ => (),
            //         }
            //     }
            //     continue 'childs;
            // }
            // };
        }
        merge_mappings(&maps, &extension)
        // vec![]
        // vec![FitMap { variable: mapping }]
    }
}

impl Symbol {
    /// Checks if the other Symbol fits into a self
    /// or it's children
    pub fn fit<'a>(&'a self, other: &'a Symbol) -> Vec<FitMap<'a>> {
        match self {
            Symbol::Variable(_) => match other {
                Symbol::Operator(_) => vec![],
                Symbol::Variable(_) => vec![FitMap {
                    variable: hashmap!(self => other),
                }],
            },
            Symbol::Operator(inner) => match other {
                Symbol::Variable(_) => vec![],
                Symbol::Operator(other) => fit_operator(&inner, &other),
            },
        }
    }

    pub fn fit_with_op<'a>(&'a self, other: &'a Operator) -> Vec<FitMap<'a>> {
        match self {
            Symbol::Variable(_) => vec![],
            Symbol::Operator(o) => fit_operator(&o, &other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_fit_root_type() {
        let a = Symbol::Variable(Variable {
            ident: String::from("a"),
        });

        let b = Symbol::Variable(Variable {
            ident: String::from("a"),
        });

        let all_maps = a.fit(&b); // Necessary to keep the vector in scope
        let map = all_maps.iter().nth(0).unwrap();
        assert_eq!(map.variable, hashmap!(&a => &b), "variable on variable");

        let a = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });

        let b = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });
        let oof = a.fit(&b);
        assert_eq!(oof.len(), 1, "operator on operator");

        let a = Symbol::Variable(Variable {
            ident: String::from("a"),
        });

        let b = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });

        assert!(a.fit(&b).is_empty(), "variable on operator");

        let a = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });

        let b = Symbol::Variable(Variable {
            ident: String::from("a"),
        });

        assert!(a.fit(&b).is_empty(), "operator on variable");
    }

    #[test]
    fn operator_flat_single_variable() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("A(b)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        let FitMap { variable } = mapping;

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
        assert_eq!(variable[&outer.childs[0]], &inner.childs[0]);
    }

    #[test]
    fn operator_flat_multiple_variables() {
        let outer = Operator::parse("A(a,b)\0");
        let inner = Operator::parse("A(c,d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 2);
        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn inner_hierarchically_variable() {
        let outer = Operator::parse("A(B(a))\0");
        let inner = Operator::parse("B(b)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 1);
        if let Symbol::Operator(ref op) = outer.childs[0] {
            assert_eq!(mapping.variable[&op.childs[0]], &inner.childs[0]);
        } else {
            assert!(false, "Expected first child to be an operator")
        };
    }

    #[test]
    fn variable_maps_to_operator() {
        let outer = Operator::parse("A(B(a))\0");
        let inner = Operator::parse("B(C(b))\0");

        // Expect a -> C(b)

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        assert_eq!(mapping.variable.len(), 1);
        let expected_key = Symbol::new("a\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let expected_value = Symbol::new("C(b)\0");
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(actual_value, &&expected_value, "Expect value to be C(b)");
        assert_eq!(format!("{}", actual_value), "C(b)");
    }

    #[test]
    fn complex_inner_simple() {
        let outer = Operator::parse("A(B(a), b)\0");
        let inner = Operator::parse("A(B(c), d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        println!("mapping: {}", format_map(mapping));
        assert_eq!(mapping.variable.len(), 2, "Expected 2 mappings");

        let expected_key = Symbol::new("a\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "c");

        let expected_key = Symbol::new("b\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable b"
        );
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "d");
    }

    #[test]
    fn complex_inner_differ_variables() {
        let outer = Operator::parse("A(B(a), a)\0");
        let inner = Operator::parse("A(B(b), c)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        println!("{:?}", mapping);

        assert!(fit_operator(&outer, &inner).is_empty());
    }

    #[test]
    fn operator_flat_multiple_variables_same_target() {
        let outer = Operator::parse("A(a,b)\0");
        let inner = Operator::parse("A(c,c)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 2);
        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn operator_flat_multiple_variables_same() {
        let outer = Operator::parse("A(a,a)\0");
        let inner = Operator::parse("A(b,b)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 1);
        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn operator_flat_multiple_variables_contradicting() {
        let outer = Operator::parse("A(a,a)\0");
        let inner = Operator::parse("A(c,d)\0");

        assert!(fit_operator(&outer, &inner).is_empty(), "Not injective");
    }

    #[test]
    fn operator_flat_wrong_childs() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("A(b,c)\0");

        assert!(fit_operator(&outer, &inner).is_empty());
    }

    #[test]
    fn operator_flat_inner_too_large() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("B(C(b))\0");

        assert!(fit_operator(&outer, &inner).is_empty());
    }

    #[test]
    fn flat_operators() {
        let outer = Symbol::new("A(a)\0");
        let inner = Symbol::new("A(b)\0");
        assert_eq!(outer.fit(&inner).len(), 1);
    }

    #[test]
    fn flat_operators_different_arg_count() {
        let outer = Symbol::new("A(a, b)\0");
        let inner = Symbol::new("A(b)\0");
        assert!(outer.fit(&inner).is_empty());
    }
}
