use super::{Context, Declaration, Rule, Symbol};
use crate::io::bag::extract_idents_from_rules;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
extern crate serde_yaml;

#[derive(Serialize, Deserialize, Default)]
struct ScenarioStringAsProblem {
    pub training: HashMap<String, String>,
    pub validation: HashMap<String, String>,
    #[serde(default)]
    pub additional_idents: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ScenarioProblemEnum {
    Inline(ScenarioStringAsProblem),
    Filename(String),
}

impl Default for ScenarioProblemEnum {
    fn default() -> Self {
        ScenarioProblemEnum::Inline(Default::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScenarioProblems {
    pub training: HashMap<String, Rule>,
    pub validation: HashMap<String, Rule>,
    pub additional_idents: Vec<String>,
}

impl ScenarioProblems {
    fn try_from_inline(
        value: &ScenarioStringAsProblem,
        parse_rule: &dyn Fn((&String, &String)) -> Vec<Result<(String, Rule), String>>,
    ) -> Result<Self, String> {
        Ok(Self {
            training: value
                .training
                .iter()
                .map(parse_rule)
                .flatten()
                .collect::<Result<HashMap<_, _>, _>>()?,
            validation: value
                .validation
                .iter()
                .map(parse_rule)
                .flatten()
                .collect::<Result<HashMap<_, _>, _>>()?,
            additional_idents: value.additional_idents.clone(),
        })
    }

    fn try_from(
        value: &ScenarioProblemEnum,
        parse_rule: &dyn Fn((&String, &String)) -> Vec<Result<(String, Rule), String>>,
    ) -> Result<Self, String> {
        match value {
            ScenarioProblemEnum::Inline(value) => {
                ScenarioProblems::try_from_inline(value, parse_rule)
            }
            ScenarioProblemEnum::Filename(filename) => Self::load(filename),
        }
    }

    pub fn load(filename: &str) -> Result<Self, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        let reader = BufReader::new(file);
        bincode::deserialize_from::<BufReader<_>, Self>(reader).map_err(|msg| msg.to_string())
    }

    pub fn dump(&self, filename: &str) -> Result<(), String> {
        let file = File::create(filename).map_err(|msg| msg.to_string())?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|msg| msg.to_string())
    }
}

#[derive(Serialize, Deserialize)]
struct ScenarioStringAsRule {
    pub declarations: HashMap<String, Declaration>,
    pub rules: HashMap<String, String>,
    #[serde(default)]
    pub problems: ScenarioProblemEnum,
    /// Used to include other yaml files content
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub premises: Vec<String>,
}

#[derive(Debug)]
pub struct Scenario {
    pub declarations: Context,
    pub rules: HashMap<String, Rule>,
    pub problems: ScenarioProblems,
    pub premises: Vec<Symbol>,
}

impl Scenario {
    pub fn load_from_yaml_reader<R>(reader: R) -> Result<Scenario, String>
    where
        R: std::io::Read + Sized,
    {
        let ss: ScenarioStringAsRule =
            serde_yaml::from_reader(reader).map_err(|msg| msg.to_string())?;

        let mut declarations = Context {
            declarations: ss.declarations,
        };
        declarations.register_standard_operators();

        let parse_rule = |(name, code): (&String, &String)| -> Vec<Result<(String, Rule), String>> {
            let postfix = vec!["", " (i)", " (ii)"];
            match Rule::parse(&declarations, code) {
                Err(msg) => vec![Err(msg)],
                Ok(rules) => {
                    let offset = if rules.len() > 1 { 1 } else { 0 };
                    rules
                        .into_iter()
                        .enumerate()
                        .map(|(i, r)| Ok((format!("{}{}", name, postfix[offset + i]), r)))
                        .collect::<Vec<_>>()
                }
            }
        };

        let rules: Result<HashMap<String, Rule>, String> =
            ss.rules.iter().map(parse_rule).flatten().collect();
        let rules = rules?;
        let problems = ScenarioProblems::try_from(&ss.problems, &parse_rule)?;

        let premises = ss
            .premises
            .iter()
            .map(|s| Symbol::parse(&declarations, s))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO: Load files from include list and merge

        Ok(Scenario {
            declarations,
            rules,
            problems,
            premises,
        })
    }

    pub fn load_from_yaml(filename: &str) -> Result<Scenario, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        Scenario::load_from_yaml_reader(file)
    }

    pub fn idents(&self, ignore_declaration: bool) -> Vec<String> {
        let mut idents = extract_idents_from_rules(
            &self.rules.iter().map(|(_, r)| r).collect::<Vec<_>>(),
            |r| r,
        );
        if !ignore_declaration {
            idents.extend(self.declarations.declarations.keys().cloned());
        }
        idents.extend(self.problems.additional_idents.iter().cloned());
        let mut idents = idents.into_iter().collect::<Vec<String>>();
        idents.sort();

        idents
    }
}

#[cfg(test)]
mod specs {
    use super::Scenario;
    use crate::context::Declaration;
    use crate::symbol::Symbol;
    use stringreader::StringReader;
    #[test]
    fn load_from_yaml_simple() {
        let reader = StringReader::new(
            r"
            declarations:
              A:
                is_fixed: true
                is_function: true
                only_root: false
            rules:
              trivial subtraction: a-a => 0
              trivial division: a/a => 1
            premises:
              - x = y
              - x = 0",
        );

        let actual = Scenario::load_from_yaml_reader(reader).unwrap();

        assert_eq!(
            actual.declarations.declarations.get("A").unwrap(),
            &Declaration {
                is_fixed: true,
                is_function: true,
                only_root: false,
            },
        );

        assert_eq!(actual.rules.len(), 2);
        assert!(actual.problems.training.is_empty());
        assert!(actual.problems.validation.is_empty());
        assert_eq!(actual.premises.len(), 2);

        assert_eq!(
            actual.premises[0],
            Symbol {
                ident: "=".to_string(),
                childs: vec![
                    Symbol::new_variable("x", false),
                    Symbol::new_variable("y", false),
                ],
                depth: 2,
                flags: 3,
                value: None,
            }
        );
    }
}
