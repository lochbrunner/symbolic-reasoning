use crate::{Rule, Symbol};
use std::collections::{HashMap, HashSet};

pub mod trace;

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct Meta {
    pub idents: Vec<String>,
    pub rule_distribution: Vec<u32>,
    /// Rule at index 0 is padding
    pub rules: Vec<Rule>,
}

// Clone is needed as long sort_map is not available
#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
pub struct FitInfo {
    /// Starting with 1 for better embedding
    pub rule_id: u32,
    pub path: Vec<usize>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct Sample {
    pub initial: Symbol,
    pub fits: Vec<FitInfo>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct SampleContainer {
    pub max_depth: u32,
    pub max_spread: u32,
    pub samples: Vec<Sample>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct Bag {
    pub meta: Meta,
    pub samples: Vec<SampleContainer>,
}

impl Bag {
    /// Iterates through all steps of all traces and puts them into a bag
    /// TODO:
    ///  * select each step and remove duplicates
    ///  * create statistics
    ///  * order by size (depth & spread)
    pub fn from_traces(traces: &[trace::Trace]) -> Bag {
        let mut rule_map: HashMap<&Rule, u32> = HashMap::new();

        let mut initials: HashMap<&Symbol, Vec<FitInfo>> = HashMap::new();
        let mut idents: HashSet<String> = HashSet::new();
        for trace in traces.iter() {
            for step in trace.all_steps() {
                let rule_id = match rule_map.get(&step.rule) {
                    None => {
                        let rule_id = rule_map.len() as u32 + 1;
                        rule_map.insert(&step.rule, rule_id);
                        rule_id
                    }
                    Some(rule_id) => *rule_id,
                };
                // Reverse rules
                let fitinfo = FitInfo {
                    path: step.path.clone(),
                    rule_id,
                };
                match initials.get_mut(&step.deduced) {
                    None => {
                        initials.insert(&step.deduced, vec![fitinfo]);
                    }
                    Some(initial) => initial.push(fitinfo),
                }
            }
            for ident in trace.meta.used_idents.iter() {
                idents.insert(ident.clone());
            }
        }
        // Sort
        let mut max_spread: u32 = 0;
        let mut max_depth = 0;
        for initial in initials.keys() {
            if max_depth < initial.depth {
                max_depth = initial.depth;
            }
            for part in initial.parts() {
                if max_spread < part.childs.len() as u32 {
                    max_spread = part.childs.len() as u32;
                }
            }
        }

        let rules = rule_map
            .iter()
            .map(|(r, _)| *r)
            .cloned()
            .collect::<Vec<_>>();
        let mut rule_distribution = vec![0; rules.len() + 1];
        for (_, fitinfos) in initials.iter() {
            for fitinfo in fitinfos.iter() {
                rule_distribution[fitinfo.rule_id as usize] += 1;
            }
        }
        rule_distribution[0] = initials.iter().fold(0, |acc, (symbol, fits)| {
            acc + ((symbol.parts().count() - 1) * fits.len()) as u32
        });

        let samples = (1..max_depth)
            .map(|depth| {
                let samples = initials
                    .iter()
                    .map(|(initial, fits)| Sample {
                        initial: (*initial).clone(),
                        fits: fits.to_vec(),
                    })
                    .collect();
                SampleContainer {
                    samples,
                    max_depth: depth,
                    max_spread,
                }
            })
            .collect();

        Bag {
            meta: Meta {
                idents: idents.into_iter().collect(),
                rules,
                rule_distribution,
            },
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
    fn create_from_flat_stages() {
        let context = Context::standard();
        let a = Symbol::parse(&context, "a").unwrap();
        let b = Symbol::parse(&context, "b").unwrap();
        let c = Symbol::parse(&context, "c").unwrap();
        let d = Symbol::parse(&context, "d").unwrap();
        let r1 = Rule::parse_first(&context, "a => b");
        let r2 = Rule::parse_first(&context, "c => d");

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

        let expected = Bag {
            meta: Meta {
                idents: vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ],
                rule_distribution: vec![0, 1, 1],
                rules: vec![r1, r2],
            },
            samples: vec![SampleContainer {
                max_depth: 1,
                max_spread: 2,
                samples: vec![
                    Sample {
                        initial: d,
                        fits: vec![FitInfo {
                            rule_id: 2,
                            path: vec![0, 0],
                        }],
                    },
                    Sample {
                        initial: b,
                        fits: vec![FitInfo {
                            rule_id: 1,
                            path: vec![0, 0],
                        }],
                    },
                ],
            }],
        };

        assert_vec_eq!(actual.meta.idents, expected.meta.idents);
        assert_vec_eq!(actual.meta.rules, expected.meta.rules);
        assert_vec_eq!(
            actual.meta.rule_distribution,
            expected.meta.rule_distribution
        );

        let actual_container = &expected.samples[0];
        let expected_container = &expected.samples[0];
        assert_eq!(actual_container.max_depth, expected_container.max_depth);
        assert_eq!(actual_container.max_spread, expected_container.max_spread);
        assert_vec_eq!(actual_container.samples, expected_container.samples);
    }
}
