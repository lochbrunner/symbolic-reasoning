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
    pub training: Vec<Rule>,
    pub validation: Vec<Rule>,
    pub additional_idents: Vec<String>,
}

impl ScenarioProblems {
    fn try_from_inline(
        value: &ScenarioStringAsProblem,
        parse_rule: &dyn Fn((&String, &String)) -> Vec<Result<Rule, String>>,
    ) -> Result<Self, String> {
        Ok(Self {
            training: value
                .training
                .iter()
                .map(parse_rule)
                .flatten()
                .collect::<Result<Vec<_>, _>>()?,
            validation: value
                .validation
                .iter()
                .map(parse_rule)
                .flatten()
                .collect::<Result<Vec<_>, _>>()?,
            additional_idents: value.additional_idents.clone(),
        })
    }

    fn try_from(
        value: &ScenarioProblemEnum,
        no_dependencies: bool,
        parse_rule: &dyn Fn((&String, &String)) -> Vec<Result<Rule, String>>,
    ) -> Result<Option<Self>, String> {
        match value {
            ScenarioProblemEnum::Inline(value) => {
                let problems = ScenarioProblems::try_from_inline(value, parse_rule)?;
                Ok(Some(problems))
            }
            ScenarioProblemEnum::Filename(filename) => {
                if no_dependencies {
                    Ok(None)
                } else {
                    let problems = Self::load(filename)?;
                    Ok(Some(problems))
                }
            }
        }
    }

    pub fn load(filename: &str) -> Result<Self, String> {
        let file = File::open(filename).map_err(|msg| format!("{}: {}", msg, filename))?;
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
    pub rules: Vec<Rule>,
    pub problems: Option<ScenarioProblems>,
    pub premises: Vec<Symbol>,
}

impl Scenario {
    pub fn load_from_yaml_reader<R>(reader: R, no_dependencies: bool) -> Result<Scenario, String>
    where
        R: std::io::Read + Sized,
    {
        let ss: ScenarioStringAsRule =
            serde_yaml::from_reader(reader).map_err(|msg| msg.to_string())?;

        let mut declarations = Context {
            declarations: ss.declarations,
        };
        declarations.register_standard_operators();

        let parse_rule = |(name, code): (&String, &String)| -> Vec<Result<Rule, String>> {
            let postfix = vec!["", " (i)", " (ii)"];
            match Rule::parse(&declarations, code) {
                Err(msg) => vec![Err(msg)],
                Ok(rules) => {
                    let offset = if rules.len() > 1 { 1 } else { 0 };
                    rules
                        .into_iter()
                        .enumerate()
                        .map(|(i, mut r)| {
                            r.name = format!("{}{}", name, postfix[offset + i]);
                            Ok(r)
                        })
                        .collect::<Vec<_>>()
                }
            }
        };

        let mut rules = ss
            .rules
            .iter()
            .map(parse_rule)
            .flatten()
            .collect::<Result<Vec<_>, _>>()?;
        // sort_by_key has lifetime issues: See
        // https://stackoverflow.com/questions/47121985/why-cant-i-use-a-key-function-that-returns-a-reference-when-sorting-a-vector-wi
        rules.sort_by(|x, y| x.name.cmp(&y.name));
        let problems = ScenarioProblems::try_from(&ss.problems, no_dependencies, &parse_rule)?;

        let premises = ss
            .premises
            .iter()
            .map(|s| Symbol::parse(&declarations, s))
            .collect::<Result<Vec<_>, _>>()?;

        for include in ss.include.iter() {
            let file = File::open(include).map_err(|msg| msg.to_string())?;
            let import = Scenario::load_from_yaml_reader(file, no_dependencies)?;
            let Scenario {
                rules: import_rules,
                declarations:
                    Context {
                        declarations: import_decls,
                    },
                ..
            } = import;
            rules.extend(import_rules);
            for (decl_name, declaration) in import_decls.into_iter() {
                declarations.declarations.insert(decl_name, declaration);
            }
        }

        Ok(Scenario {
            declarations,
            rules,
            problems,
            premises,
        })
    }

    pub fn load_from_yaml(filename: &str, no_dependencies: bool) -> Result<Scenario, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        Scenario::load_from_yaml_reader(file, no_dependencies)
    }

    pub fn idents(&self, ignore_declaration: bool) -> Vec<String> {
        let mut idents = extract_idents_from_rules(&self.rules, |r| r);
        if !ignore_declaration {
            idents.extend(self.declarations.declarations.keys().cloned());
        }
        if let Some(ref problems) = self.problems {
            idents.extend(problems.additional_idents.iter().cloned());
        }
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

        let actual = Scenario::load_from_yaml_reader(reader, false).unwrap();

        assert_eq!(
            actual.declarations.declarations.get("A").unwrap(),
            &Declaration {
                is_fixed: true,
                is_function: true,
                only_root: false,
            },
        );

        assert_eq!(actual.rules.len(), 2);
        assert!(actual.problems.as_ref().unwrap().training.is_empty());
        assert!(actual.problems.as_ref().unwrap().validation.is_empty());
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
