use std::fmt;
use std::str;

use super::parsers;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Variable {
    pub ident: String,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Constant {
    pub ident: String,
}

impl Variable {}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.ident)
    }
}

/// Operator and Function are for the sake of simplicity
/// the same
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Operator {
    pub ident: String,
    pub childs: Vec<Symbol>,
    /// The depth of the hierarchal structure
    pub depth: u32,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let body: Vec<String> = self.childs.iter().map(|c| c.to_string()).collect();
        write!(f, "{}({})", self.ident, body.join(","))
    }
}

impl Operator {
    pub fn parse(s: &str) -> Operator {
        parsers::parse_operator(s).unwrap().1
    }

    pub fn calc_depth(childs: &Vec<Symbol>) -> u32 {
        let max_child = childs
            .iter()
            .map(|c| match c {
                Symbol::Operator(o) => o.depth,
                Symbol::Variable(_) => 1,
            })
            .max();

        match max_child {
            Some(c) => c + 1,
            None => 1,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Symbol {
    Variable(Variable),
    Operator(Operator),
    // Constant(Constant),
}

impl Symbol {
    pub fn new(s: &str) -> Symbol {
        parsers::parse_symbol(s).unwrap().1
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
    use crate::*;

    #[test]
    fn variable_fmt() {
        let v = Variable {
            ident: String::from("a"),
        };
        assert_eq!(format!("{}", v), "a");
    }

    #[test]
    fn operator_fmt() {
        let a = Symbol::Variable(Variable {
            ident: String::from("a"),
        });
        let b = Symbol::Variable(Variable {
            ident: String::from("b"),
        });
        let o = Operator {
            ident: String::from("f"),
            depth: 0,
            childs: vec![a, b],
        };
        assert_eq!(format!("{}", o), "f(a,b)");
    }
}
