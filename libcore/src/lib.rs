//! Rules are used to transform one expression into another
//!
//! # Example
//!
//! ```latex
//! A*B+A*C -> A*(B+C)
//! ```
//!
//! Can be applied to
//!
//! ```latex
//! r*s+r*t
//! ```
//!
//! which transforms it to
//!
//! ```latex
//! r*(s+t)
//! ```

#[macro_use]
extern crate nom;

use std::fmt;
use std::str;

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
        write!(f, "{}({})", self.ident, body.join(", "))
    }
}

#[derive(Debug, PartialEq)]
pub enum Symbol {
    Variable(Variable),
    Operator(Operator),
}

impl Symbol {
    pub fn new(s: &str) -> Symbol {
        symbol(s).unwrap().1
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

named!(
    array<&str,Vec<Symbol>>,
    ws!(delimited!(
        char!('('),
        separated_list!(char!(','), symbol),
        char!(')')
    ))
);

named!(
    operator<&str,Operator>,
    do_parse!(
        ident: string
            >> childs: array
            >> (Operator {
                ident: String::from(ident),
                childs
            })
    )
);

fn is_alphabetic_c(chr: char) -> bool {
    (chr as u8 >= 0x41 && chr as u8 <= 0x5A) || (chr as u8 >= 0x61 && chr as u8 <= 0x7A)
}

named!(
    string<&str, &str>,
        escaped!(take_while1!(is_alphabetic_c), '\\', one_of!("\"n\\"))
);

named!(
    symbol<&str,Symbol>,
    ws!(alt!(
      operator => {|o| Symbol::Operator(o)} |
      string =>   {|s| Symbol::Variable(Variable{ident: String::from(s)}) }
    ))
);

#[derive(Debug, PartialEq)]
pub struct Rule {
    pub condition: Symbol,
    pub conclusion: Symbol,
}

named!(
    rule<&str, Rule>,
    ws!(
        do_parse!(
            condition: symbol >>
            tag!("=>") >>
            conclusion: symbol >>
            (Rule{condition, conclusion})
        )
    )
);

impl Rule {
    pub fn new(code: &str) -> Rule {
        rule(code).unwrap().1
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} => {}", self.condition, self.conclusion)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn parse_symbol() {
        assert_eq!(
            Symbol::new("A(a,b,c)\0"),
            Symbol::Operator(Operator {
                ident: String::from("A"),
                childs: vec![
                    Symbol::Variable(Variable {
                        ident: String::from("a")
                    }),
                    Symbol::Variable(Variable {
                        ident: String::from("b")
                    }),
                    Symbol::Variable(Variable {
                        ident: String::from("c")
                    })
                ]
            })
        );

        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn operator_e2e() {
        let op = Symbol::new("A(a,b,c)\0");
        assert_eq!(op.to_string(), "A(a, b, c)");
    }

    #[test]
    fn rule_e2e_variable() {
        let rule = Rule::new("a  => a\0");
        assert_eq!(rule.to_string(), "a => a")
    }

    #[test]
    fn rule_e2e_operator() {
        let rule = Rule::new("A(a,b)  => B(c,d)\0");
        assert_eq!(rule.to_string(), "A(a, b) => B(c, d)")
    }
}
