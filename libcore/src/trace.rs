use super::{Rule, Symbol};
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

#[derive(Serialize)]
pub struct Trace<'a> {
    pub initial: &'a Symbol,
    pub stage: Vec<TraceStep<'a>>,
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

#[derive(Deserialize, Serialize)]
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
    pub initial: Symbol,
    pub stage: Vec<DenseTraceStep>,
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
            initial: trace.initial.clone(),
            stage: trace
                .stage
                .iter()
                .map(|ts| DenseTrace::from_trace_step(ts))
                .collect(),
        }
    }
}

impl DenseTrace {
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
