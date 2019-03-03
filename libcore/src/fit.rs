use super::symbol::*;

use maplit::*;
use std::collections::HashMap;

/// Use this struct later
#[derive(Debug)]
pub struct FitMap<'a> {
    // TODO: Needs to be public?
    pub variable: HashMap<&'a Symbol, &'a Symbol>,
    // location: &'a Symbol,
}

fn fit_operator_flat<'a>(outer: &'a Operator, inner: &'a Operator) -> Vec<FitMap<'a>> {
    let mut mapping = HashMap::new();
    // Check root
    if outer.ident != inner.ident {
        // Outer must be larger
        // if outer.depth > inner.depth {
        // Try fit the childs of the outer
        // Maybe not the fastest implementation?
        outer
            .childs
            .iter()
            .fold(Vec::<FitMap<'a>>::new(), |mut acc, child| {
                acc.extend(child.fit_with_op(&inner));
                acc
            })
    } else if outer.childs.len() != inner.childs.len() {
        // Wrong number of childs
        // Is it expected that this could happen?
        vec![]
    } else {
        // Check for variable repetition
        for i in 0..outer.childs.len() {
            // Check for injective
            if mapping.contains_key(&outer.childs[i]) {
                // Do we still have this mapping?
                if mapping.get(&outer.childs[i]).unwrap() == &&inner.childs[i] {
                    continue;
                }
                return vec![];
            }
            // Check for surjective
            mapping.insert(&outer.childs[i], &inner.childs[i]);
        }
        vec![FitMap { variable: mapping }]
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
                Symbol::Operator(other) => fit_operator_flat(&inner, &other),
            },
        }
    }

    pub fn fit_with_op<'a>(&'a self, other: &'a Operator) -> Vec<FitMap<'a>> {
        match self {
            Symbol::Variable(_) => vec![],
            Symbol::Operator(o) => fit_operator_flat(&o, &other),
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
        let all_mappings = fit_operator_flat(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable.len(), 1);

        assert_eq!(format!("{}", mapping.variable.keys().nth(0).unwrap()), "a");
        assert_eq!(
            format!("{}", mapping.variable.values().nth(0).unwrap()),
            "b"
        );
    }

    #[test]
    fn operator_flat_multiple_variables() {
        let outer = Operator::parse("A(a,b)\0");
        let inner = Operator::parse("A(c,d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator_flat(&outer, &inner);
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
        let all_mappings = fit_operator_flat(&outer, &inner);
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
        let all_mappings = fit_operator_flat(&outer, &inner);
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
    fn complex_inner() {
        let outer = Operator::parse("A(B(a), b)\0");
        let inner = Operator::parse("A(B(c), d)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator_flat(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        assert_eq!(mapping.variable.len(), 2);

        let expected_key = Symbol::new("B(a)\0");
        assert!(
            mapping.variable.contains_key(&expected_key),
            "Expect mapping contains variable a"
        );
        let actual_value = mapping.variable.get(&expected_key).unwrap();
        assert_eq!(format!("{}", actual_value), "B(c)");

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
        let all_mappings = fit_operator_flat(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();
        println!("{:?}", mapping);

        assert!(fit_operator_flat(&outer, &inner).is_empty());
    }

    #[test]
    fn operator_flat_multiple_variables_same_target() {
        let outer = Operator::parse("A(a,b)\0");
        let inner = Operator::parse("A(c,c)\0");

        // Necessary to keep the vector in scope
        let all_mappings = fit_operator_flat(&outer, &inner);
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
        let all_mappings = fit_operator_flat(&outer, &inner);
        let mapping = all_mappings.iter().nth(0).unwrap();

        assert_eq!(mapping.variable.len(), 1);
        assert_eq!(mapping.variable[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.variable[&outer.childs[1]], &inner.childs[1]);
    }

    #[test]
    fn operator_flat_multiple_variables_contradicting() {
        let outer = Operator::parse("A(a,a)\0");
        let inner = Operator::parse("A(c,d)\0");

        assert!(
            fit_operator_flat(&outer, &inner).is_empty(),
            "Not injective"
        );
    }

    #[test]
    fn operator_flat_wrong_childs() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("A(b,c)\0");

        assert!(fit_operator_flat(&outer, &inner).is_empty());
    }

    #[test]
    fn operator_flat_inner_too_large() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("B(C(b))\0");

        assert!(fit_operator_flat(&outer, &inner).is_empty());
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
