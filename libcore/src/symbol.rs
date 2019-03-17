use std::fmt;
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

    pub fn new_operator(ident: &str, childs: Vec<Symbol>) -> Symbol {
        Symbol {
            ident: String::from(ident),
            depth: Symbol::calc_depth(&childs),
            fixed: ident.chars().nth(0).unwrap().is_uppercase(),
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

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.childs.len() {
            0 => write!(f, "{}", self.ident),
            _ => {
                let body: Vec<String> = self.childs.iter().map(|c| c.to_string()).collect();
                write!(f, "{}({})", self.ident, body.join(","))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn variable_fmt() {
        let v = Symbol::new_variable("a");
        assert_eq!(format!("{}", v), "a");
    }

    #[test]
    fn operator_fmt() {
        let a = Symbol::new_variable("a");
        let b = Symbol::new_variable("b");
        let o = Symbol::new_operator("f", vec![a, b]);
        assert_eq!(format!("{}", o), "f(a,b)");
    }
}
