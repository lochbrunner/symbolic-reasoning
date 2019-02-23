use std::fmt;
use std::str;

use super::parsers;

#[derive(Debug, PartialEq)]
pub struct Variable {
    pub ident: String,
}

impl Variable {}

/// Operator and Function are for the sake of simplicity
/// the same
#[derive(Debug, PartialEq)]
pub struct Operator {
    pub ident: String,
    pub childs: Vec<Symbol>,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let body: Vec<String> = self.childs.iter().map(|c| c.to_string()).collect();
        write!(f, "{}({})", self.ident, body.join(","))
    }
}

#[derive(Debug, PartialEq)]
pub enum Symbol {
    Variable(Variable),
    Operator(Operator),
}

impl Symbol {
    pub fn new(s: &str) -> Symbol {
        parsers::parse_symbol(s).unwrap().1
    }
    /// Checks if the other Symbol fits into a self
    /// or it's children
    pub fn fit(self, other: &Symbol) -> bool {
        match self {
            Symbol::Variable(_) => match other {
                Symbol::Operator(_) => false,
                Symbol::Variable(_) => true,
            },
            Symbol::Operator(_) => false,
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Symbol::Variable(v) => write!(f, "{}", v.ident),
            Symbol::Operator(o) => write!(f, "{}", o),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn symbol_fit() {
        assert_eq!(3, 3);
    }
}
