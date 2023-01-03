use super::{Context, Symbol};

use std::fmt;

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rule {
    pub condition: Symbol,
    pub conclusion: Symbol,
    pub name: String,
}

impl Default for Rule {
    fn default() -> Self {
        Self {
            condition: Default::default(),
            conclusion: Default::default(),
            name: "Padding".to_owned(),
        }
    }
}

impl Rule {
    #[cfg(test)]
    pub fn parse_first(context: &Context, code: &str) -> Rule {
        let mut parts = code.split("=>").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion")).unwrap();
            let condition = Symbol::parse(context, parts.pop().expect("Condition")).unwrap();
            Rule {
                conclusion,
                condition,
                name: "test rule".to_owned(),
            }
        } else {
            panic!();
        }
    }
    // TODO: return multiple rules when finding <=>
    pub fn parse(context: &Context, code: &str) -> Result<Vec<Rule>, String> {
        let mut parts = code.split("<=>").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(vec![
                Rule {
                    conclusion: condition.clone(),
                    condition: conclusion.clone(),
                    name: "(i)".to_owned(),
                },
                Rule {
                    conclusion,
                    condition,
                    name: "(iI)".to_owned(),
                },
            ]);
        }
        let mut parts = code.split("=>").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(vec![Rule {
                conclusion,
                condition,
                name: "".to_owned(),
            }]);
        }
        let mut parts = code.split("<=").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            return Ok(vec![Rule {
                conclusion,
                condition,
                name: "".to_owned(),
            }]);
        }
        let mut parts = code.split(":=").collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(vec![Rule {
                conclusion,
                condition,
                name: "".to_owned(),
            }]);
        }

        let mut parts = code.split('=').collect::<Vec<&str>>();
        if parts.len() == 2 {
            let conclusion = Symbol::parse(context, parts.pop().expect("Conclusion"))?;
            let condition = Symbol::parse(context, parts.pop().expect("Condition"))?;
            return Ok(vec![Rule {
                conclusion,
                condition,
                name: "".to_owned(),
            }]);
        }

        Err(format!("Can not parse {} as rule", code))
    }

    /// Returns a Rule with swapped condition <-> conclusion
    pub fn reverse(&self) -> Rule {
        Rule {
            condition: self.conclusion.clone(),
            conclusion: self.condition.clone(),
            name: self.name.clone(),
        }
    }
}

impl fmt::Display for Rule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} => {}", self.condition, self.conclusion)
    }
}
