use std::str;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Symbol {
    pub ident: String,
    // Empty for non operators
    pub childs: Vec<Symbol>,
    /// The depth of the hierarchal structure
    pub depth: u32,
    /// For now specifies, if a variable of function is a concrete,
    /// Starting with uppercase means true, else false.
    /// commonly known constant/function or not.
    /// Specifies if this symbol should be mapped or not.
    /// Example function: "+" Example variable constant of gravitation "G"
    /// Later on it will depend on the context if function is fixed or not.
    /// For instance "G" could stand for another variable, but not the constant of gravitation.
    pub fixed: bool,
}

impl Symbol {
    pub fn new_variable(ident: &str) -> Symbol {
        Symbol {
            ident: String::from(ident),
            depth: 1,
            fixed: ident.chars().nth(0).unwrap().is_uppercase(),
            childs: Vec::new(),
        }
    }

    pub fn new_variable_from_string(ident: String) -> Symbol {
        Symbol {
            fixed: ident.chars().nth(0).unwrap().is_uppercase(),
            ident: ident,
            depth: 1,
            childs: Vec::new(),
        }
    }

    pub fn new_operator(ident: &str, childs: Vec<Symbol>) -> Symbol {
        Symbol {
            ident: String::from(ident),
            depth: Symbol::calc_depth(&childs),
            fixed: ident.chars().nth(0).unwrap().is_uppercase(),
            childs,
        }
    }

    pub fn new_operator_from_string(ident: String, childs: Vec<Symbol>) -> Symbol {
        Symbol {
            fixed: ident.chars().nth(0).unwrap().is_uppercase(),
            ident,
            depth: Symbol::calc_depth(&childs),
            childs,
        }
    }

    pub fn calc_depth(childs: &Vec<Symbol>) -> u32 {
        let max_child = childs.iter().map(|c| c.depth).max();

        match max_child {
            Some(c) => c + 1,
            None => 1,
        }
    }
}
