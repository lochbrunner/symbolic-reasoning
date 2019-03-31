use super::parsers_dep::parse_rule;
use super::symbol;

use std::fmt;

#[derive(Debug, PartialEq)]
pub struct Rule {
    pub condition: symbol::Symbol,
    pub conclusion: symbol::Symbol,
}

impl Rule {
    pub fn parse(code: &str) -> Rule {
        parse_rule(code).unwrap().1
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} => {}", self.condition, self.conclusion)
    }
}
