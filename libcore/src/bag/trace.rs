use crate::{Rule, Symbol};
use crate::dumper::latex::LaTeX;
use std::collections::HashSet;
extern crate chrono;
extern crate serde_yaml;

#[derive(Serialize, Clone, PartialEq, Eq, Hash)]
pub struct ApplyInfo<'a> {
    pub rule: &'a Rule,
    pub path: Vec<usize>,
    pub initial: Symbol,
    pub deduced: Symbol,
}

impl<'a> ApplyInfo<'a> {
    #[allow(dead_code)]
    pub fn print_header() {
        println!("  {0: <14} | {1: <14} | {2: <14}", "new", "initial", "rule");
        println!("  -----------------------------------------");
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        let ded_str = format!("{}", self.deduced);
        let ini_str = format!("{}", self.initial);
        let rule = format!("{}", self.rule);
        println!("  {0: <14} | {1: <14} | {2: <14}", ded_str, ini_str, rule);
    }
}

#[derive(Serialize, PartialEq, Eq, Hash)]
pub struct TraceStep<'a> {
    pub info: ApplyInfo<'a>,
    pub successors: Vec<TraceStep<'a>>,
}

#[derive(Deserialize, Serialize, Clone, Default)]
pub struct Meta {
    pub used_idents: HashSet<String>,
    pub rules: Vec<Rule>,
}

#[derive(Serialize)]
pub struct Trace<'a> {
    pub meta: Meta,
    pub initial: &'a Symbol,
    pub stages: Vec<TraceStep<'a>>,
}

impl<'a> Trace<'a> {
    pub fn write_bincode<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        match bincode::serialize_into(writer, self) {
            Ok(_) => Ok(()),
            Err(msg) => Err(msg.to_string()),
        }
    }

    pub fn write_yaml<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        match serde_yaml::to_writer(writer, self) {
            Ok(_) => Ok(()),
            Err(msg) => Err(msg.to_string()),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct DenseApplyInfo {
    pub rule: Rule,
    pub path: Vec<usize>,
    pub initial: Symbol,
    pub deduced: Symbol,
}

#[derive(Deserialize, Serialize)]
pub struct DenseTraceStep {
    pub info: DenseApplyInfo,
    pub successors: Vec<DenseTraceStep>,
}

#[derive(Deserialize, Serialize)]
pub struct DenseTrace {
    pub meta: Meta,
    pub initial: Symbol,
    pub stages: Vec<DenseTraceStep>,
}

impl DenseTrace {
    fn from_apply_info(apply_info: &ApplyInfo) -> DenseApplyInfo {
        DenseApplyInfo {
            rule: (*apply_info.rule).clone(),
            path: apply_info.path.clone(),
            initial: apply_info.initial.clone(),
            deduced: apply_info.deduced.clone(),
        }
    }

    fn from_trace_step(trace_step: &TraceStep) -> DenseTraceStep {
        DenseTraceStep {
            info: DenseTrace::from_apply_info(&trace_step.info),
            successors: trace_step
                .successors
                .iter()
                .map(|ts| DenseTrace::from_trace_step(ts))
                .collect(),
        }
    }

    pub fn from_trace(trace: &Trace) -> DenseTrace {
        DenseTrace {
            meta: trace.meta.clone(),
            initial: trace.initial.clone(),
            stages: trace
                .stages
                .iter()
                .map(|ts| DenseTrace::from_trace_step(ts))
                .collect(),
        }
    }
}

pub struct Calculation<'a> {
    pub steps: Vec<&'a DenseApplyInfo>,
}

pub struct TraceIter<'a> {
    cursor: Vec<usize>,
    trace: &'a DenseTrace,
}

impl<'a> TraceIter<'a> {
    #[inline]
    fn get_steps(&self) -> Vec<&'a DenseApplyInfo> {
        // Extract item
        let mut step = &self.trace.stages;
        let mut steps = vec![];
        for i in self.cursor.iter() {
            steps.push(&step[*i].info);
            step = &step[*i].successors;
        }
        steps
    }

    /// Returns an error, if it was not possible to go one step sideward
    #[inline]
    fn try_go_sideward(&mut self) -> Result<(), ()> {
        if self.cursor.is_empty() {
            return Err(());
        }
        // Check we can go sideward
        let mut current_stage = &self.trace.stages;
        // Go to second last
        if self.cursor.len() > 1 {
            for i in self.cursor.iter().take(self.cursor.len() - 1) {
                current_stage = &current_stage[*i].successors;
            }
        }
        if current_stage.len() > 1 + *self.cursor.last().unwrap() {
            *self.cursor.last_mut().unwrap() += 1;
            Ok(())
        } else {
            // Indicate to go one up to the next: Recursion
            Err(())
        }
    }

    #[inline]
    fn go_to_ground(&mut self) {
        let mut current_stage = &self.trace.stages;
        for i in self.cursor.iter() {
            current_stage = &current_stage[*i].successors;
        }

        while !current_stage.is_empty() {
            self.cursor.push(0);
            current_stage = &current_stage[0].successors;
        }
    }
}

impl<'a> Iterator for TraceIter<'a> {
    type Item = Calculation<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Starting with an empty vector
        if self.cursor.is_empty() {
            None
        } else {
            let steps = self.get_steps();
            while let Err(_) = self.try_go_sideward() {
                // Go upwards
                if self.cursor.is_empty() {
                    break;
                } else {
                    self.cursor.pop();
                }
            }
            // empty cursor indicates end of tree
            if !self.cursor.is_empty() {
                self.go_to_ground();
            }
            Some(Calculation { steps })
        }
    }
}

pub struct StepsIter<'a> {
    cursors: Vec<Vec<usize>>,
    trace: &'a DenseTrace,
}

impl<'a> StepsIter<'a> {
    fn get_node(&self, cursor: &[usize]) -> &'a DenseTraceStep {
        let mut current_stage = &self.trace.stages;
        for i in cursor.iter().take(cursor.len() - 1) {
            current_stage = &current_stage[*i].successors;
        }
        &current_stage[*cursor.last().unwrap()]
    }
}

impl<'a> Iterator for StepsIter<'a> {
    type Item = &'a DenseApplyInfo;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.cursors.pop() {
            None => None,
            Some(cursor) => {
                let node = self.get_node(&cursor);
                for (i, _) in node.successors.iter().enumerate() {
                    let mut cursor = cursor.clone();
                    cursor.push(i);
                    self.cursors.push(cursor);
                }
                Some(&node.info)
            }
        }
    }
}

impl DenseTrace {
    pub fn initial_cursor(&self) -> Vec<usize> {
        let mut cursor = Vec::new();
        let mut current_stage = &self.stages;

        while !current_stage.is_empty() {
            cursor.push(0);
            current_stage = &current_stage[0].successors;
        }
        cursor
    }
    pub fn unroll(&self) -> TraceIter {
        TraceIter {
            cursor: self.initial_cursor(),
            trace: self,
        }
    }

    pub fn all_steps(&self) -> StepsIter {
        StepsIter {
            cursors: self
                .stages
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i])
                .collect(),
            trace: self,
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

    pub fn write_yaml<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        match serde_yaml::to_writer(writer, self) {
            Ok(_) => Ok(()),
            Err(msg) => Err(msg.to_string()),
        }
    }

    pub fn write_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        writeln!(writer, "\\documentclass{{scrartcl}}\n")?;
        writeln!(writer, "\\usepackage[utf8]{{inputenc}}")?;
        writeln!(writer, "\\usepackage[T1]{{fontenc}}")?;
        writeln!(writer, "\\usepackage{{lmodern}}")?;
        writeln!(writer, "\\usepackage[ngerman]{{babel}}")?;
        writeln!(writer, "\\usepackage{{amsmath}}\n")?;
        writeln!(writer, "\\usepackage{{xcolor}}\n")?;

        writeln!(writer, "\\title{{Calculation}}")?;
        writeln!(writer, "\\author{{Matthias Lochbrunner}}")?;
        writeln!(
            writer,
            "\\date{{{}}}",
            chrono::Utc::now().format("%a %b %e %Y")
        )?;
        writeln!(writer, "\\begin{{document}}\n")?;
        writeln!(writer, "\\maketitle")?;
        writeln!(writer, "\\tableofcontents")?;

        // Calculation
        writeln!(writer, "\\section{{Scenarios}}\n")?;
        writeln!(writer, "Initial term")?;
        writeln!(writer, "\\begin{{flalign*}}")?;
        self.initial.writeln_latex(writer)?;
        writeln!(writer, "\\end{{flalign*}}")?;

        for (i, calculation) in self.unroll().enumerate().take(20) {
            writeln!(writer, "\\subsection{{Calculation {}}}\n", i + 1)?;
            writeln!(writer, "\\begin{{flalign*}}")?;
            for step in calculation.steps.iter() {
                step.rule.write_latex(writer)?;
                write!(writer, "&\\implies ")?;
                step.deduced.write_latex_highlight(&step.path, writer)?;
                writeln!(writer, "\\\\[\\parskip]")?;
            }
            writeln!(writer, "\\end{{flalign*}}")?;
        }

        // End of document
        writeln!(writer, "\\end{{document}}")?;

        Ok(())
    }

    pub fn read_bincode<R>(reader: R) -> Result<DenseTrace, String>
    where
        R: std::io::Read,
    {
        let r = bincode::deserialize_from::<R, DenseTrace>(reader);
        match r {
            Ok(trace) => Ok(trace),
            Err(msg) => Err(msg.to_string()),
        }
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::context::Context;

    #[test]
    fn rollout_no_stages() {
        let context = Context::standard();
        let trace = DenseTrace {
            meta: Meta::default(),
            stages: vec![],
            initial: Symbol::parse(&context, "a"),
        };

        let lines = trace.unroll().collect::<Vec<Calculation>>();

        assert!(lines.is_empty());
    }

    #[test]
    fn rollout_flat_stages() {
        let context = Context::standard();

        let stages = [Symbol::parse(&context, "a"), Symbol::parse(&context, "b")]
            .into_iter()
            .map(|deduced| DenseTraceStep {
                info: DenseApplyInfo {
                    rule: Rule::parse(&context, "v => c"),
                    path: vec![],
                    initial: Symbol::parse(&context, "i"),
                    deduced: deduced.clone(),
                },
                successors: vec![],
            })
            .collect();

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let lines = trace.unroll().collect::<Vec<Calculation>>();

        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn rollout_hierarchical_two_stages() {
        let context = Context::standard();

        let get_stage = |symbols: Vec<(&'static str, Vec<DenseTraceStep>)>| -> Vec<DenseTraceStep> {
            symbols
                .into_iter()
                .map(|(symbol, successors)| DenseTraceStep {
                    info: DenseApplyInfo {
                        rule: Rule::parse(&context, "v => c"),
                        path: vec![],
                        initial: Symbol::parse(&context, "i"),
                        deduced: Symbol::parse(&context, symbol),
                    },
                    successors,
                })
                .collect()
        };

        let stage_1 = get_stage(vec![("a", vec![]), ("b", vec![])]);
        let stage_2 = get_stage(vec![("c", vec![]), ("d", vec![])]);

        let stages = get_stage(vec![("v", stage_1), ("u", stage_2)]);

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let lines = trace
            .unroll()
            .map(|l| l.steps.last().unwrap().deduced.ident.clone())
            .collect::<Vec<String>>();

        assert_eq!(lines.len(), 4);

        let expected = ["a", "b", "c", "d"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        assert_eq!(lines, expected);
    }

    #[test]
    fn rollout_hierarchical_tree_stages() {
        let context = Context::standard();

        let get_stage = |symbols: Vec<(&'static str, Vec<DenseTraceStep>)>| -> Vec<DenseTraceStep> {
            symbols
                .into_iter()
                .map(|(symbol, successors)| DenseTraceStep {
                    info: DenseApplyInfo {
                        rule: Rule::parse(&context, "v => c"),
                        path: vec![],
                        initial: Symbol::parse(&context, "i"),
                        deduced: Symbol::parse(&context, symbol),
                    },
                    successors,
                })
                .collect()
        };

        let stage_a_1 = get_stage(vec![("a", vec![]), ("b", vec![])]);
        let stage_a_2 = get_stage(vec![("c", vec![]), ("d", vec![])]);
        let stage_b_1 = get_stage(vec![("e", vec![]), ("f", vec![])]);
        let stage_b_2 = get_stage(vec![("g", vec![]), ("h", vec![])]);

        let stage_a = get_stage(vec![("A", stage_a_1), ("B", stage_a_2)]);
        let stage_b = get_stage(vec![("C", stage_b_1), ("D", stage_b_2)]);

        let stages = get_stage(vec![("v", stage_a), ("u", stage_b)]);

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let lines = trace
            .unroll()
            .map(|l| l.steps.last().unwrap().deduced.ident.clone())
            .collect::<Vec<String>>();

        assert_eq!(lines.len(), 8);

        let expected = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        assert_eq!(lines, expected);
    }

    #[test]
    fn all_steps_no_stages() {
        let context = Context::standard();
        let trace = DenseTrace {
            meta: Meta::default(),
            stages: vec![],
            initial: Symbol::parse(&context, "a"),
        };

        let lines = trace.all_steps().collect::<Vec<&DenseApplyInfo>>();

        assert!(lines.is_empty());
    }

    #[test]
    fn all_steps_flat_stages() {
        let context = Context::standard();

        let stages = [Symbol::parse(&context, "a"), Symbol::parse(&context, "b")]
            .into_iter()
            .map(|deduced| DenseTraceStep {
                info: DenseApplyInfo {
                    rule: Rule::parse(&context, "v => c"),
                    path: vec![],
                    initial: Symbol::parse(&context, "i"),
                    deduced: deduced.clone(),
                },
                successors: vec![],
            })
            .collect();

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let steps = trace
            .all_steps()
            .map(|l| l.deduced.ident.clone())
            .collect::<Vec<String>>();;

        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn all_steps_hierarchical_two_stages() {
        let context = Context::standard();

        let get_stage = |symbols: Vec<(&'static str, Vec<DenseTraceStep>)>| -> Vec<DenseTraceStep> {
            symbols
                .into_iter()
                .map(|(symbol, successors)| DenseTraceStep {
                    info: DenseApplyInfo {
                        rule: Rule::parse(&context, "v => c"),
                        path: vec![],
                        initial: Symbol::parse(&context, "i"),
                        deduced: Symbol::parse(&context, symbol),
                    },
                    successors,
                })
                .collect()
        };

        let stage_1 = get_stage(vec![("a", vec![]), ("b", vec![])]);
        let stage_2 = get_stage(vec![("c", vec![]), ("d", vec![])]);

        let stages = get_stage(vec![("v", stage_1), ("u", stage_2)]);

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let steps = trace
            .all_steps()
            .map(|l| l.deduced.ident.clone())
            .collect::<Vec<String>>();

        assert_eq!(steps.len(), 6);

        // pre order. Childs in reverse order
        let expected = ["u", "d", "c", "v", "b", "a"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        assert_eq!(steps, expected);
    }

    #[test]
    fn all_steps_hierarchical_tree_stages() {
        let context = Context::standard();

        let get_stage = |symbols: Vec<(&'static str, Vec<DenseTraceStep>)>| -> Vec<DenseTraceStep> {
            symbols
                .into_iter()
                .map(|(symbol, successors)| DenseTraceStep {
                    info: DenseApplyInfo {
                        rule: Rule::parse(&context, "v => c"),
                        path: vec![],
                        initial: Symbol::parse(&context, "i"),
                        deduced: Symbol::parse(&context, symbol),
                    },
                    successors,
                })
                .collect()
        };

        let stage_a_1 = get_stage(vec![("a", vec![]), ("b", vec![])]);
        let stage_a_2 = get_stage(vec![("c", vec![]), ("d", vec![])]);
        let stage_b_1 = get_stage(vec![("e", vec![]), ("f", vec![])]);
        let stage_b_2 = get_stage(vec![("g", vec![]), ("h", vec![])]);

        let stage_a = get_stage(vec![("A", stage_a_1), ("B", stage_a_2)]);
        let stage_b = get_stage(vec![("C", stage_b_1), ("D", stage_b_2)]);

        let stages = get_stage(vec![("v", stage_a), ("u", stage_b)]);

        let trace = DenseTrace {
            meta: Meta::default(),
            stages,
            initial: Symbol::parse(&context, "a"),
        };

        let steps = trace
            .all_steps()
            .map(|l| l.deduced.ident.clone())
            .collect::<Vec<String>>();

        p!(steps);

        let expected = [
            "u", "D", "h", "g", "C", "f", "e", "v", "B", "d", "c", "A", "b", "a",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

        assert_eq!(steps.len(), expected.len());
        assert_eq!(steps, expected);
    }
}
