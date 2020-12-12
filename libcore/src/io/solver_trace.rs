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

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TraceStatistics {
    pub success: bool,
    pub fit_tries: u32,
    pub fit_results: u32,
    pub trace: StepInfo,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProblemStatistics {
    pub problem_name: String,
    pub iterations: Vec<TraceStatistics>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ProblemSummary {
    pub problem_name: String,
    pub success: bool,
}

impl ProblemStatistics {
    pub fn summary(&self) -> ProblemSummary {
        ProblemSummary {
            problem_name: self.problem_name.clone(),
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
