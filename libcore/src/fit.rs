use super::symbol::*;

use maplit::*;
use std::collections::HashMap;

fn fit_operator_flat<'a>(
    outer: &'a Operator,
    inner: &'a Operator,
) -> Result<HashMap<&'a Symbol, &'a Symbol>, &'static str> {
    let mut mapping = HashMap::new();
    // Check root
    if outer.ident != inner.ident {
        // Outer must be larger
        if outer.depth > inner.depth {
            // Return list
            for child in outer.childs.iter() {
                let _e: Result<HashMap<&'a Symbol, &'a Symbol>, &'static str> =
                    match child.fit_with_op(&inner) {
                        _ => Err("Not implemented yet error"),
                    };
            }
            Err("Empty array")
        } else {
            Err("Inner operator is larger than outer operator.")
        }
    } else if outer.childs.len() != inner.childs.len() {
        Err("Wrong number of childs")
    } else {
        // Check for variable repetition
        for i in 0..outer.childs.len() {
            mapping.insert(&outer.childs[i], &inner.childs[i]);
        }
        Ok(mapping)
    }
}

impl Symbol {
    /// Checks if the other Symbol fits into a self
    /// or it's children
    pub fn fit<'a>(
        &'a self,
        other: &'a Symbol,
    ) -> Result<HashMap<&'a Symbol, &'a Symbol>, &'static str> {
        match self {
            Symbol::Variable(_) => match other {
                Symbol::Operator(_) => Err(""),
                Symbol::Variable(_) => Ok(hashmap!(self => other)),
            },
            Symbol::Operator(inner) => match other {
                Symbol::Variable(_) => Err(""),
                Symbol::Operator(other) => fit_operator_flat(&inner, &other),
            },
        }
    }

    pub fn fit_with_op<'a>(
        &'a self,
        other: &'a Operator,
    ) -> Result<HashMap<&'a Symbol, &'a Symbol>, &'static str> {
        match self {
            Symbol::Variable(_) => Err(""),
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
        assert_eq!(a.fit(&b), Ok(hashmap!(&a => &b)));

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
        assert!(a.fit(&b).is_ok());

        let a = Symbol::Variable(Variable {
            ident: String::from("a"),
        });

        let b = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });
        assert!(!a.fit(&b).is_ok());

        let a = Symbol::Operator(Operator {
            ident: String::from("a"),
            depth: 1,
            childs: Vec::new(),
        });

        let b = Symbol::Variable(Variable {
            ident: String::from("a"),
        });
        assert!(!a.fit(&b).is_ok());
    }

    #[test]
    fn fit_operator_flat_simple() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("A(b)\0");

        let mapping = fit_operator_flat(&outer, &inner).unwrap();

        assert_eq!(mapping[&outer.childs[0]], &inner.childs[0]);
        assert_eq!(mapping.len(), 1);

        assert_eq!(format!("{}", mapping.keys().nth(0).unwrap()), "a");
        assert_eq!(format!("{}", mapping.values().nth(0).unwrap()), "b");
    }

    #[test]
    fn fit_operator_flat_wrong_childs() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("A(b,c)\0");

        assert_eq!(
            fit_operator_flat(&outer, &inner),
            Err("Wrong number of childs")
        );
    }

    #[test]
    fn fit_operator_flat_inner_too_large() {
        let outer = Operator::parse("A(a)\0");
        let inner = Operator::parse("B(C(b))\0");

        assert_eq!(
            fit_operator_flat(&outer, &inner),
            Err("Inner operator is larger than outer operator.")
        );
    }

    #[test]
    fn symbol_fit_flat_operators() {
        let outer = Symbol::new("A(a)\0");
        let inner = Symbol::new("A(b)\0");
        assert!(outer.fit(&inner).is_ok())
    }

    #[test]
    fn symbol_dont_fit_flat_operators() {
        let outer = Symbol::new("A(a, b)\0");
        let inner = Symbol::new("A(b)\0");
        assert!(!outer.fit(&inner).is_ok())
    }
}
