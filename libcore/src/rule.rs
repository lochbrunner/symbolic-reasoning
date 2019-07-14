// use super::parsers_dep::parse_rule;
use super::{Context, Symbol};

use std::fmt;

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rule {
    pub condition: Symbol,
    pub conclusion: Symbol,
}

impl Rule {
    pub fn parse(context: &Context, code: &str) -> Result<Rule, String> {
        let mut parts = code.split("=>").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(Rule {
                conclusion,
                condition,
            });
        }
        let mut parts = code.split(":=").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(Rule {
                conclusion,
                condition,
            });
        }

        let mut parts = code.split('=').collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(Rule {
                conclusion,
                condition,
            });
        }

        Err(format!("Can not parse {} as rule", code))
    }

    /// Returns a Rule with swapped condition <-> conclusion
    pub fn reverse(&self) -> Rule {
        Rule {
            condition: self.conclusion.clone(),
            conclusion: self.condition.clone(),
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} => {}", self.condition, self.conclusion)
    }
}
