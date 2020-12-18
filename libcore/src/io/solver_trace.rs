use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Later merge from different iterations in order to compare them
#[derive(Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct StepInfo {
    pub current_latex: String,
    pub value: Option<f32>,
    pub confidence: Option<f32>,
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
struct StepInfoCombinedRef {
    pub current: String,
    pub subsequent: Vec<String>,
}

// static none_apply: Option<ApplyInfoCombined> = None;

impl StepInfoCombined {
    fn create(single: &StepInfo, size: usize) -> Self {
        Self {
            current_latex: single.current_latex.clone(),
            contributed: false,
            subsequent: vec![],
            rule_id: single.rule_id,
            path: single.path.clone(),
            iterations: vec![None; size],
        }
    }
}

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
            iterations,
            success: self.iterations.iter().any(|it| it.success),
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct SolverStatistics {
    pub problems: HashMap<String, ProblemStatistics>,
}

impl SolverStatistics {
    pub fn summaries(&self) -> Vec<ProblemSummary> {
        self.problems.values().map(|p| p.summary()).collect()
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
