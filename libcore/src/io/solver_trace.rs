use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Later merge from different iterations in order to compare them
#[derive(Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct StepInfo {
    pub current_latex: String,
    // TODO: abort_reason ?
    pub value: Option<f32>,
    pub confidence: Option<f32>,
    // How to handle multiple ways leading to the same term?
    // Vec<enum{StepInfo, String}> ?
    pub subsequent: Vec<StepInfo>,
    pub rule_id: u32,
    pub path: Vec<usize>,
    pub top: u32,
    pub contributed: bool,
}

impl StepInfo {
    pub fn max_depth(&self) -> u32 {
        self.subsequent
            .iter()
            .fold(0, |prev, curr| std::cmp::max(prev, curr.max_depth()))
            + 1
    }

    pub fn depth_of_solution(&self) -> Option<u32> {
        if self.contributed {
            Some(
                self.subsequent
                    .iter()
                    .map(|s| s.depth_of_solution())
                    .filter(|d| d.is_some())
                    .fold(0, |acc, d| std::cmp::max(acc, d.unwrap()))
                    + 1,
            )
        } else {
            None
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TraceStatistics {
    pub success: bool,
    pub fit_tries: u32,
    pub fit_results: u32,
    pub trace: StepInfo,
}

impl TraceStatistics {
    pub fn max_depth(&self) -> u32 {
        self.trace.max_depth()
    }

    pub fn depth_of_solution(&self) -> Option<u32> {
        self.trace.depth_of_solution()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProblemStatistics {
    pub problem_name: String,
    pub iterations: Vec<TraceStatistics>,
    pub target_latex: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ApplyInfoCombined {
    pub value: Option<f32>,
    pub confidence: Option<f32>,
    pub top: u32,
}

#[derive(Serialize, Deserialize)]
pub struct StepInfoCombined {
    pub current_latex: String,
    pub contributed: bool,
    pub subsequent: Vec<StepInfoCombined>,
    pub rule_id: u32,
    pub path: Vec<usize>,
    pub iterations: Vec<Option<ApplyInfoCombined>>,
}

/// Use this to track the node's references
// struct StepInfoCombinedRef {
//     pub current: String,
//     pub subsequent: Vec<String>,
// }

// static none_apply: Option<ApplyInfoCombined> = None;

// impl StepInfoCombined {
//     fn create(single: &StepInfo, size: usize) -> Self {
//         Self {
//             current_latex: single.current_latex.clone(),
//             contributed: false,
//             subsequent: vec![],
//             rule_id: single.rule_id,
//             path: single.path.clone(),
//             iterations: vec![None; size],
//         }
//     }
// }

#[derive(Serialize, Deserialize)]
pub struct ProblemStatisticsMetaCombined {
    pub success: bool,
    pub fit_tries: u32,
    pub fit_results: u32,
}

#[derive(Serialize, Deserialize)]
pub struct ProblemStatisticsCombined {
    pub meta: Vec<ProblemStatisticsMetaCombined>,
    pub problem_name: String,
    pub trace: StepInfoCombined,
}

impl From<&ProblemStatistics> for ProblemStatisticsCombined {
    fn from(source: &ProblemStatistics) -> Self {
        assert!(
            source.iterations.len() > 0,
            "Expects at least one iteration!"
        );
        // let first = &source.iterations[0].trace;
        // let trace = StepInfoCombined::create(first, source.iterations.len());
        // let mut root = StepInfoCombinedRef {
        //     current: first.current_latex.clone(),
        //     subsequent: vec![],
        // };

        // let mut step_map: HashMap<String, StepInfoCombined> = HashMap::new();
        // step_map.insert(trace.current_latex.clone(), trace);

        // for (index, iteration) in source.iterations.iter().enumerate() {
        //     let trace = &iteration.trace;
        //     if let Some(ref mut target) = step_map.get_mut(&trace.current_latex) {
        //         target.contributed |= trace.contributed;
        //         target.iterations[index] = Some(ApplyInfoCombined {
        //             value: trace.value,
        //             confidence: trace.confidence,
        //             top: trace.top,
        //         });
        //         for child in trace.subsequent.iter() {
        //             if let None = step_map.get(&child.current_latex) {
        //                 let cc = StepInfoCombined::create(first, source.iterations.len());
        //                 step_map.insert(child.current_latex.clone(), cc);
        //                 // target.subsequent.push(cc);
        //             }
        //         }
        //     }
        // }

        unimplemented!();
        // Use a HashMap for each node
        // Self {
        //     problem_name: orig.problem_name,
        //     meta: vec![],
        //     trace: StepInfoCombined {},
        // }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct IterationSummary {
    pub fit_results: u32,
    pub max_depth: u32,
    pub depth_of_solution: Option<u32>,
    pub success: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ProblemSummary {
    pub name: String,
    pub initial_latex: Option<String>,
    pub target_latex: String,
    pub iterations: Vec<IterationSummary>,
    pub success: bool,
}

impl ProblemStatistics {
    pub fn summary(&self) -> ProblemSummary {
        let iterations = self
            .iterations
            .iter()
            .map(|it| IterationSummary {
                fit_results: it.fit_results,
                success: it.success,
                max_depth: it.max_depth(),
                depth_of_solution: it.depth_of_solution(),
            })
            .collect();
        let initial_latex = self
            .iterations
            .get(0)
            .and_then(|it| Some(it.trace.current_latex.clone()));
        ProblemSummary {
            name: self.problem_name.clone(),
            initial_latex,
            target_latex: self.target_latex.clone(),
            iterations,
            success: self.iterations.iter().any(|it| it.success),
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct SolverStatistics {
    pub problems: Vec<ProblemStatistics>,
    pub name_map: Option<HashMap<String, usize>>,
    pub name_index: Option<Vec<usize>>,
    pub success_index: Option<Vec<usize>>,
    pub fits_index: Option<Vec<usize>>,
    pub solution_depth_index: Option<Vec<usize>>,
    pub total_depth_index: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SortingKey {
    None,
    Name,
    Success,
    Fits,
    SolutionDepth,
    TotalDepth,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SingleSummaryQuery {
    pub key: SortingKey,
    pub up: bool,
    pub index: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BatchSummaryQuery {
    pub key: SortingKey,
    pub up: bool,
    pub begin: usize,
    pub end: usize,
}

impl SolverStatistics {
    pub fn summaries(&self) -> Vec<ProblemSummary> {
        self.problems.iter().map(|p| p.summary()).collect()
    }

    pub fn get_problem<'a>(&'a self, name: &str) -> Result<&'a ProblemStatistics, String> {
        let name_map = self.name_map.as_ref().ok_or("No index created yet!")?;
        let index = name_map
            .get(name)
            .ok_or(format!("Problem with name {} is not available!", name))?;
        Ok(&self.problems[*index])
    }

    pub fn create_index<E>(
        &mut self,
        progress_reporter: &dyn Fn(f32) -> Result<(), E>,
    ) -> Result<(), E> {
        // Six steps
        let step_size = 1. / 6.;

        let mut name_map: HashMap<String, usize> = HashMap::new();
        for (i, problem) in self.problems.iter().enumerate() {
            name_map.insert(problem.problem_name.clone(), i);
        }
        self.name_map = Some(name_map);
        progress_reporter(step_size)?;

        // Name index
        let mut name_maps = self
            .problems
            .iter()
            .map(|s| &s.problem_name)
            .enumerate()
            .collect::<Vec<_>>();

        name_maps.sort_by_key(|(_, n)| *n);

        self.name_index = Some(name_maps.iter().map(|(i, _)| *i).collect());
        progress_reporter(step_size * 2.)?;

        let summaries = self.summaries();

        // Fit maps
        let mut fits_maps = summaries
            .iter()
            .map(|s| {
                s.iterations
                    .iter()
                    .map(|i| i.fit_results)
                    .max()
                    .unwrap_or_default()
            })
            .enumerate()
            .collect::<Vec<_>>();

        fits_maps.sort_by_key(|(_, f)| *f);

        self.fits_index = Some(fits_maps.iter().map(|(i, _)| *i).collect());
        progress_reporter(step_size * 3.)?;

        // Solution depth
        let mut sd_maps = summaries
            .iter()
            .map(|s| {
                s.iterations
                    .iter()
                    .map(|i| i.depth_of_solution.unwrap_or(0))
                    .max()
                    .unwrap_or_default()
            })
            .enumerate()
            .collect::<Vec<_>>();

        sd_maps.sort_by_key(|(_, d)| *d);
        self.solution_depth_index = Some(sd_maps.iter().map(|(i, _)| *i).collect());
        progress_reporter(step_size * 4.)?;

        // Total depth
        let mut td_maps = summaries
            .iter()
            .map(|s| {
                s.iterations
                    .iter()
                    .map(|i| i.max_depth)
                    .max()
                    .unwrap_or_default()
            })
            .enumerate()
            .collect::<Vec<_>>();

        td_maps.sort_by_key(|(_, d)| *d);
        self.total_depth_index = Some(td_maps.iter().map(|(i, _)| *i).collect());
        progress_reporter(step_size * 5.)?;

        // Success index
        let mut success_maps = summaries
            .iter()
            .map(|s| {
                s.iterations
                    .iter()
                    .map(|i| if i.success { 1 } else { 0 })
                    .max()
                    .unwrap_or_default()
            })
            .enumerate()
            .collect::<Vec<_>>();

        success_maps.sort_by_key(|(_, s)| *s);
        self.success_index = Some(success_maps.iter().map(|(i, _)| *i).collect());
        progress_reporter(step_size * 6.)?;
        Ok(())
    }

    pub fn query_summary(&self, query: &SingleSummaryQuery) -> Result<ProblemSummary, String> {
        let index = if query.up {
            query.index
        } else {
            self.problems.len() - 1 - query.index
        };
        let index = match query.key {
            SortingKey::None => &query.index,
            SortingKey::Name => self
                .name_index
                .as_ref()
                .ok_or(format!("Fit index not available!"))?
                .get(index)
                .ok_or(format!("Could not find item #{} in name index", index))?,
            SortingKey::Success => self
                .success_index
                .as_ref()
                .ok_or(format!("Fit index not available!"))?
                .get(index)
                .ok_or(format!("Could not find item #{} in success index", index))?,
            SortingKey::Fits => self
                .fits_index
                .as_ref()
                .ok_or(format!("Fit index not available!"))?
                .get(index)
                .ok_or(format!("Could not find item #{} in fitting index", index))?,
            SortingKey::SolutionDepth => self
                .solution_depth_index
                .as_ref()
                .ok_or(format!("Fit index not available!"))?
                .get(index)
                .ok_or(format!(
                    "Could not find item #{} in solution depth index",
                    index
                ))?,
            SortingKey::TotalDepth => self
                .total_depth_index
                .as_ref()
                .ok_or(format!("Fit index not available!"))?
                .get(index)
                .ok_or(format!(
                    "Could not find item #{} in total depth index",
                    index
                ))?,
        };
        Ok(self.problems[*index].summary())
    }

    pub fn query_summaries(
        &self,
        query: &BatchSummaryQuery,
    ) -> Result<Vec<ProblemSummary>, String> {
        let BatchSummaryQuery {
            key,
            up,
            begin,
            end,
        } = query.clone();
        if begin >= self.len() {
            Err(format!(
                "Begin {} is larger than container length {}",
                begin,
                self.len()
            ))
        } else {
            let end = if end < self.len() { end } else { self.len() };
            (begin..end)
                .map(|index| {
                    self.query_summary(&SingleSummaryQuery {
                        index,
                        up,
                        key: key.clone(),
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        }
    }

    pub fn len(&self) -> usize {
        self.problems.len()
    }

    pub fn write_bincode<W>(&self, writer: W) -> Result<(), String>
    where
        W: std::io::Write,
    {
        bincode::serialize_into(writer, self).map_err(|msg| msg.to_string())
    }

    pub fn read_bincode<R>(reader: R) -> Result<Self, String>
    where
        R: std::io::Read,
    {
        bincode::deserialize_from::<R, Self>(reader).map_err(|msg| msg.to_string())
    }

    pub fn dump(&self, filename: &str) -> Result<(), String> {
        let file =
            File::create(filename.clone()).map_err(|msg| format!("{}: \"{}\"", msg, filename))?;
        let writer = BufWriter::new(file);
        self.write_bincode(writer)
    }

    pub fn load(filename: &str) -> Result<Self, String> {
        let file =
            File::open(filename.clone()).map_err(|msg| format!("{}: \"{}\"", msg, filename))?;
        let reader = BufReader::new(file);
        Self::read_bincode(reader)
    }
}

#[cfg(test)]
mod specs {
    use super::*;

    #[test]
    fn query_fit_results() {
        let mut stat = SolverStatistics::default();
        stat.problems.push(ProblemStatistics {
            problem_name: "p1".to_owned(),
            iterations: vec![TraceStatistics {
                success: true,
                fit_tries: 4,
                fit_results: 4,
                trace: StepInfo::default(),
            }],
            target_latex: "t1".to_owned(),
        });
        stat.problems.push(ProblemStatistics {
            problem_name: "p2".to_owned(),
            iterations: vec![TraceStatistics {
                success: false,
                fit_tries: 3,
                fit_results: 3,
                trace: StepInfo::default(),
            }],
            target_latex: "t2".to_owned(),
        });
        stat.problems.push(ProblemStatistics {
            problem_name: "p3".to_owned(),
            iterations: vec![TraceStatistics {
                success: false,
                fit_tries: 2,
                fit_results: 2,
                trace: StepInfo::default(),
            }],
            target_latex: "t3".to_owned(),
        });
        stat.problems.push(ProblemStatistics {
            problem_name: "p4".to_owned(),
            iterations: vec![TraceStatistics {
                success: true,
                fit_tries: 1,
                fit_results: 1,
                trace: StepInfo::default(),
            }],
            target_latex: "t4".to_owned(),
        });

        assert!(stat.fits_index.is_none());

        assert!(stat.create_index::<()>(&|_| Ok(())).is_ok());

        assert!(stat.fits_index.is_some());

        let response = stat.query_summaries(&BatchSummaryQuery {
            key: SortingKey::Fits,
            up: true,
            begin: 0,
            end: 4,
        });

        assert!(response.is_ok());
        let response = response.unwrap();

        let fits = response
            .iter()
            .map(|s| s.iterations[0].fit_results)
            .collect::<Vec<_>>();

        assert_eq!(&fits, &[1, 2, 3, 4]);

        let response = stat.query_summaries(&BatchSummaryQuery {
            key: SortingKey::Fits,
            up: false,
            begin: 0,
            end: 4,
        });

        assert!(response.is_ok());
        let response = response.unwrap();

        let fits = response
            .iter()
            .map(|s| s.iterations[0].fit_results)
            .collect::<Vec<_>>();

        assert_eq!(&fits, &[4, 3, 2, 1]);
    }
}
