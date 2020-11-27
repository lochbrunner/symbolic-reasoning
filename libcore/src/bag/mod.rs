use crate::scenario::Scenario;
use crate::{Rule, Symbol};
use std::collections::{HashMap, HashSet};

pub mod fitinfo;
pub use fitinfo::{FitCompare, FitInfo, Policy};
pub mod trace;

pub fn extract_idents_from_rules<T>(rules: &[T], unpack: fn(&T) -> &Rule) -> HashSet<String> {
    let mut used_idents = HashSet::new();

    for rule in rules.iter().map(unpack) {
        for part in rule
            .condition
            .parts()
            .filter(|s| s.fixed())
            .map(|s| &s.ident)
        {
            if !used_idents.contains(part) {
                used_idents.insert(part.clone());
            }
        }

        for part in rule
            .conclusion
            .parts()
            .filter(|s| s.fixed())
            .map(|s| &s.ident)
        {
            if !used_idents.contains(part) {
                used_idents.insert(part.clone());
            }
        }
    }

    used_idents
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Clone)]
pub struct Meta {
    pub idents: Vec<String>,
    pub rule_distribution: Vec<(u32, u32)>,
    /// contributed, not contributed
    pub value_distribution: (u32, u32),
    /// Rule at index 0 is padding
    pub rules: Vec<(String, Rule)>,
    // Should we add the scenario file name?
}

impl Meta {
    pub fn from_scenario(scenario: &Scenario) -> Self {
        let mut rules: Vec<(String, Rule)> = Vec::new();
        rules.push(("padding".to_string(), Default::default()));
        let scenario_rules = scenario
            .rules
            .iter()
            .map(|(k, v)| (k.clone(), v.reverse()))
            .collect::<Vec<_>>();
        rules.extend_from_slice(&scenario_rules);
        let idents = extract_idents_from_rules(&rules, |(_, r)| r)
            .iter()
            .cloned()
            .collect();
        Self {
            rule_distribution: vec![(1, 1); rules.len()],
            value_distribution: (0, 0),
            rules,
            idents,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct Sample {
    pub initial: Symbol,
    /// For value network
    pub useful: bool,
    pub fits: Vec<FitInfo>,
}

/// TODO:
///  * make sure that Samples are unique -> Use HashMap
///  * Method to merge containers
///  * Move functionality from pycore to core
#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
pub struct SampleContainer {
    pub max_depth: u32,
    pub max_spread: u32,
    pub max_size: u32,
    pub samples: Vec<Sample>,
}

#[derive(Deserialize, Serialize, Debug, PartialEq)]
pub struct Bag {
    pub meta: Meta,
    pub containers: Vec<SampleContainer>,
}

impl Bag {
    /// Iterates through all steps of all traces and puts them into a bag
    /// TODO:
    ///  * select each step and remove duplicates
    ///  * create statistics
    pub fn from_traces<'a>(
        traces: &'a [trace::Trace],
        filter: &dyn Fn(&Symbol) -> bool,
        augment: &dyn Fn((&Symbol, FitInfo)) -> Vec<(Symbol, FitInfo)>,
    ) -> Self {
        let mut rule_map: HashMap<&Rule, u32> = HashMap::new();
        let mut rules: Vec<(String, Rule)> = Vec::new();
        rules.push(("padding".to_string(), Default::default()));

        let mut initials: HashMap<Symbol, Vec<FitInfo>> = HashMap::new();
        let mut idents: HashSet<String> = HashSet::new();
        for trace in traces.iter() {
            for ident in trace.meta.used_idents.iter() {
                idents.insert(ident.clone());
            }
            for (name, rule) in trace.meta.rules.iter() {
                if !rule_map.contains_key(&rule) {
                    rule_map.insert(&rule, rules.len() as u32);
                    rules.push((name.to_string(), rule.clone()));
                }
            }
            for (deduced, fitinfo) in trace
                .all_steps()
                .filter(|s| filter(&s.deduced))
                .map(|step| {
                    let rule_id = *rule_map.get(&step.rule).expect("Rule");
                    (
                        &step.deduced,
                        FitInfo {
                            path: step.path.clone(),
                            rule_id,
                            policy: Policy::Positive,
                        },
                    )
                })
                .map(augment)
                .flatten()
            {
                // Reverse rules
                match initials.get_mut(&deduced) {
                    None => {
                        initials.insert(deduced, vec![fitinfo]);
                    }
                    Some(initial) => initial.push(fitinfo),
                }
            }
        }
        // Sort
        let mut max_spread: u32 = 0;
        let mut max_depth = 0;
        let mut max_size: u32 = 0;
        for initial in initials.keys() {
            if max_depth < initial.depth {
                max_depth = initial.depth;
            }
            let size = initial.size();
            if max_size < size {
                max_size = size;
            }
            for part in initial.parts() {
                if max_spread < part.childs.len() as u32 {
                    max_spread = part.childs.len() as u32;
                }
            }
        }

        let mut rule_distribution = vec![(0, 0); rules.len()];
        // Assuming each rule was useful
        let mut value_distribution = (0, 0);
        for (_, fitinfos) in initials.iter() {
            for fitinfo in fitinfos.iter() {
                let (ref mut positive, ref mut negative) =
                    rule_distribution[fitinfo.rule_id as usize];
                if fitinfo.policy == Policy::Positive {
                    *positive += 1;
                    let (ref mut positive_value, _) = value_distribution;
                    *positive_value += 1;
                } else {
                    *negative += 1;
                }
            }
        }
        rule_distribution[0] = initials
            .iter()
            .fold((0, 0), |(positive, _), (symbol, fits)| {
                (
                    positive + ((symbol.parts().count() - 1) * fits.len()) as u32,
                    0,
                )
            });

        let containers = (1..max_depth)
            .map(|depth| {
                let samples = initials
                    .iter()
                    .filter(|(initial, _)| initial.depth == depth)
                    .map(|(initial, fits)| Sample {
                        initial: (*initial).clone(),
                        useful: true,
                        fits: fits.to_vec(),
                    })
                    .collect();
                SampleContainer {
                    samples,
                    max_depth: depth,
                    max_spread,
                    max_size,
                }
            })
            .collect();

        Bag {
            meta: Meta {
                idents: idents.into_iter().collect(),
                rules,
                rule_distribution,
                value_distribution,
            },
            containers,
        }
    }

    /// Creates a bag with one empty container
    pub fn empty(max_spread: u32, scenario: &Scenario) -> Self {
        Bag {
            meta: Meta::from_scenario(scenario),
            containers: vec![SampleContainer {
                max_size: 0,
                max_depth: 0,
                max_spread,
                samples: vec![],
            }],
        }
    }

    pub fn write_bincode<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        bincode::serialize_into(writer, self).map_err(|msg| msg.to_string())
    }

    pub fn read_bincode<R>(reader: R) -> Result<Bag, String>
    where
        R: std::io::Read,
    {
        bincode::deserialize_from::<R, Bag>(reader).map_err(|msg| msg.to_string())
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
        let r1 = ("r1".to_string(), Rule::parse_first(&context, "a => b"));
        let r2 = ("r2".to_string(), Rule::parse_first(&context, "c => d"));
        let pad_rule = ("padding".to_string(), Default::default());

        let traces = vec![
            Trace {
                meta: trace::Meta {
                    used_idents: hashset! {"a".to_string(), "b".to_string()},
                    rules: vec![r1.clone()],
                },
                initial: &a,
                stages: vec![TraceStep {
                    info: ApplyInfo {
                        rule: &r1.1,
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
                        rule: &r2.1,
                        path: vec![0, 0],
                        initial: c.clone(),
                        deduced: d.clone(),
                    },
                    successors: vec![],
                }],
            },
        ];

        let actual = Bag::from_traces(&traces, &|_| true, &|s| vec![(s.0.clone(), s.1)]);

        let expected = Bag {
            meta: Meta {
                idents: vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ],
                rule_distribution: vec![(0, 0), (1, 0), (1, 0)],
                value_distribution: (2, 0),
                rules: vec![pad_rule, r1, r2],
            },
            containers: vec![SampleContainer {
                max_depth: 1,
                max_spread: 2,
                max_size: 1,
                samples: vec![
                    Sample {
                        initial: d,
                        fits: vec![FitInfo {
                            rule_id: 2,
                            path: vec![0, 0],
                            policy: Policy::Positive,
                        }],
                        useful: true,
                    },
                    Sample {
                        initial: b,
                        fits: vec![FitInfo {
                            rule_id: 1,
                            path: vec![0, 0],
                            policy: Policy::Positive,
                        }],
                        useful: true,
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

        let actual_container = &expected.containers[0];
        let expected_container = &expected.containers[0];
        assert_eq!(actual_container.max_depth, expected_container.max_depth);
        assert_eq!(actual_container.max_spread, expected_container.max_spread);
        assert_vec_eq!(actual_container.samples, expected_container.samples);
    }

    #[test]
    fn fits_compare() {
        let a = FitInfo {
            policy: Policy::Negative,
            rule_id: 0,
            path: vec![0, 1],
        };

        let same = FitInfo {
            policy: Policy::Negative,
            rule_id: 0,
            path: vec![0, 1],
        };

        assert_eq!(a.compare(&same), FitCompare::Matching);

        let other_rule = FitInfo {
            policy: Policy::Negative,
            rule_id: 1,
            path: vec![0, 1],
        };

        assert_eq!(a.compare(&other_rule), FitCompare::Unrelated);

        let other_path = FitInfo {
            policy: Policy::Negative,
            rule_id: 0,
            path: vec![1, 0],
        };

        assert_eq!(a.compare(&other_path), FitCompare::Unrelated);

        let contradicting = FitInfo {
            policy: Policy::Positive,
            rule_id: 0,
            path: vec![0, 1],
        };

        assert_eq!(a.compare(&contradicting), FitCompare::Contradicting);
    }

    #[test]
    fn fits_compare_many() {
        let a = FitInfo {
            policy: Policy::Negative,
            rule_id: 0,
            path: vec![0, 1],
        };

        let others = [
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 1,
                path: vec![0, 1],
            },
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 0,
                path: vec![1, 0],
            },
        ];

        assert_eq!(a.compare_many(&others, |f| f), FitCompare::Unrelated);

        let one_matching = [
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 1,
                path: vec![0, 1],
            },
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 0,
                path: vec![1, 0],
            },
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 0,
                path: vec![0, 1],
            },
        ];

        assert_eq!(a.compare_many(&one_matching, |f| f), FitCompare::Matching);

        let one_contradicting = [
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 1,
                path: vec![0, 1],
            },
            &FitInfo {
                policy: Policy::Negative,
                rule_id: 0,
                path: vec![1, 0],
            },
            &FitInfo {
                policy: Policy::Positive,
                rule_id: 0,
                path: vec![0, 1],
            },
        ];

        assert_eq!(
            a.compare_many(&one_contradicting, |f| f),
            FitCompare::Contradicting
        );
    }
}
