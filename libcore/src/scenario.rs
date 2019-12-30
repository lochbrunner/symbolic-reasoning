use super::{Context, Declaration, Rule, Symbol};
use std::collections::HashMap;
use std::fs::File;
extern crate serde_yaml;

#[derive(Serialize, Deserialize)]
struct ScenarioStringAsRule {
    pub declarations: HashMap<String, Declaration>,
    pub rules: HashMap<String, String>,
    #[serde(default)]
    pub problems: HashMap<String, String>,
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
    pub problems: HashMap<String, Rule>,
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
            match Rule::parse(&declarations, code) {
                Err(msg) => vec![Err(msg)],
                Ok(rules) => rules
                    .into_iter()
                    .map(|r| Ok((name.to_string(), r)))
                    .collect::<Vec<_>>(),
            }
        };

        let rules: Result<HashMap<String, Rule>, String> =
            ss.rules.iter().map(parse_rule).flatten().collect();
        let rules = rules?;
        let problems: Result<HashMap<String, Rule>, String> =
            ss.problems.iter().map(parse_rule).flatten().collect();
        let problems = problems?;

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
        assert!(actual.problems.is_empty());
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
