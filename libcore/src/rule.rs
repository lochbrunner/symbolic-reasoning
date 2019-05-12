// use super::parsers_dep::parse_rule;
use super::symbol::Symbol;
use crate::context::Context;

use std::fmt;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Rule {
    pub condition: Symbol,
    pub conclusion: Symbol,
}

impl Rule {
    pub fn parse(context: &Context, code: &str) -> Rule {
        let mut parts = code.split("=>").collect::<Vec<&str>>();
        Rule {
            conclusion: Symbol::parse(context, parts.pop().expect("Conclusion")),
            condition: Symbol::parse(context, parts.pop().expect("Condition")),
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} => {}", self.condition, self.conclusion)
    }
}
