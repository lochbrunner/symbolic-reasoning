use crate::fit::fit;
use crate::{Rule, Symbol};
use std::collections::{HashMap, HashSet};

pub mod trace;

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct RuleStatistics {
    pub rule: Rule,
    // Number of fit resulting that rule
    pub fits: usize,
    /// How many times it was good to use this rule
    pub viably: usize,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Meta {
    pub idents: Vec<String>,
    pub rules: Vec<RuleStatistics>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct FitInfo {
    pub rule: Rule,
    pub path: Vec<usize>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct Sample {
    pub initial: Symbol,
    pub fits: Vec<FitInfo>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Bag {
    pub meta: Meta,
    pub samples: Vec<Sample>,
}

impl Bag {
    /// Iterates through all steps of all traces and tries to fit all rules
    pub fn from_traces(traces: &[trace::Trace]) -> Bag {
        let mut idents: HashSet<String> = HashSet::new();
        let mut rules: HashMap<&Rule, RuleStatistics> = HashMap::new();

        // Add all rules in advanced in order to remove bias leter
        for trace in traces.iter() {
            // add idents
            idents.extend(trace.meta.used_idents.iter().cloned());
            // add rules
            for rule in trace.meta.rules.iter() {
                match rules.get_mut(rule) {
                    None => {
                        rules.insert(
                            rule,
                            RuleStatistics {
                                rule: rule.reverse(),
                                viably: 0,
                                fits: 0,
                            },
                        );
                    }
                    Some(_) => (),
                }
            }
        }

        let mut samples: Vec<Sample> = Vec::new();
        let mut seen_initials: HashSet<&Symbol> = HashSet::new();

        for trace in traces.iter() {
            // add samples + rule statistics
            let mut stages = trace.stages.iter().collect::<Vec<&trace::TraceStep>>();
            'stages: loop {
                match stages.pop() {
                    None => break 'stages,
                    Some(stage) => {
                        // Use reversed rule
                        let initial = &stage.info.deduced;
                        if !seen_initials.contains(initial) {
                            seen_initials.insert(initial);
                            let mut fits = vec![];
                            for rule in rules.iter_mut() {
                                let rule_fits = fit(&initial, &rule.0.conclusion);
                                let rule_fits = rule_fits
                                    .into_iter()
                                    .map(|f| FitInfo {
                                        rule: (*rule.0).reverse(),
                                        path: f.path,
                                    })
                                    .collect::<Vec<FitInfo>>();
                                rule.1.fits += rule_fits.len();
                                fits.extend(rule_fits);
                            }
                            samples.push(Sample {
                                initial: initial.clone(),
                                fits,
                            });
                        }
                        stages.extend(stage.successors.iter());
                    }
                }
            }
        }
        // Convert to vector
        let rules = rules.into_iter().map(|s| s.1).collect();
        let idents = idents.into_iter().collect();
        Bag {
            meta: Meta { idents, rules },
            samples,
        }
    }

    pub fn write_bincode<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        match bincode::serialize_into(writer, self) {
            Ok(_) => Ok(()),
            Err(msg) => Err(msg.to_string()),
        }
    }

    pub fn read_bincode<R>(reader: R) -> Result<Bag, String>
    where
        R: std::io::Read,
    {
        let bag = bincode::deserialize_from::<R, Bag>(reader);
        match bag {
            Ok(bag) => Ok(bag),
            Err(msg) => Err(msg.to_string()),
        }
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::context::Context;
    use trace::{ApplyInfo, Trace, TraceStep};

    #[test]
    fn from_traces() {
        let context = Context::standard();
        let a = Symbol::parse(&context, "a");
        let b = Symbol::parse(&context, "b");
        let c = Symbol::parse(&context, "c");
        let d = Symbol::parse(&context, "d");
        let r1 = Rule::parse(&context, "a => b");
        let r2 = Rule::parse(&context, "c => d");

        let traces = vec![
            Trace {
                meta: trace::Meta {
                    used_idents: hashset! {"a".to_string(), "b".to_string()},
                    rules: vec![r1.clone()],
                },
                initial: &a,
                stages: vec![TraceStep {
                    info: ApplyInfo {
                        rule: &r1,
                        path: vec![0, 0],
                        initial: a.clone(),
                        deduced: b.clone(),
                    },
                    successors: vec![],
                }],
            },
            Trace {
                meta: trace::Meta {
                    used_idents: hashset! {"c".to_string(), "d".to_string()},
                    rules: vec![r2.clone()],
                },
                initial: &c,
                stages: vec![TraceStep {
                    info: ApplyInfo {
                        rule: &r2,
                        path: vec![0, 0],
                        initial: c.clone(),
                        deduced: d.clone(),
                    },
                    successors: vec![],
                }],
            },
        ];

        let actual = Bag::from_traces(&traces);

        let expected_idents = ["a", "b", "c", "d"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let mut actual_idents = actual.meta.idents;
        actual_idents.sort();

        assert_eq!(actual_idents, expected_idents);

        let expected_rules = vec![
            RuleStatistics {
                rule: r1.reverse(),
                fits: 2,
                viably: 0,
            },
            RuleStatistics {
                rule: r2.reverse(),
                fits: 2,
                viably: 0,
            },
        ];

        // Sort the actual in order to have a deterministic outcome
        let mut actual_rules = actual.meta.rules;
        actual_rules.sort_by(|l, r| {
            if r == l {
                std::cmp::Ordering::Equal
            } else if l.rule == r1 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        assert_eq!(actual_rules, expected_rules);

        let expected_samples = vec![
            Sample {
                initial: b.clone(),
                fits: vec![
                    FitInfo {
                        rule: r1.reverse(),
                        path: vec![],
                    },
                    FitInfo {
                        rule: r2.reverse(),
                        path: vec![],
                    },
                ],
            },
            Sample {
                initial: d.clone(),
                fits: vec![
                    FitInfo {
                        rule: r1.reverse(),
                        path: vec![],
                    },
                    FitInfo {
                        rule: r2.reverse(),
                        path: vec![],
                    },
                ],
            },
        ];

        assert_eq!(actual.samples, expected_samples);
    }
}
