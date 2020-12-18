//! This module contains containers to store tracing results of the solver.
//! Especially for the t3_loop.py script.
//!
use crate::common::op_to_string;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{FileNotFoundError, IOError, KeyError, NotImplementedError, ReferenceError};
use pyo3::prelude::*;
use pyo3::PyMappingProtocol;
use pyo3::PyNumberProtocol;
use pyo3::PyObjectProtocol;
use std::sync::Arc;

use core::io::solver_trace::{
    IterationSummary, ProblemStatistics, ProblemSummary, SolverStatistics, StepInfo,
    TraceStatistics,
};

fn get_mut<'a, T>(reference: &'a mut Arc<T>) -> PyResult<&'a mut T> {
    Arc::get_mut(reference).ok_or(PyErr::new::<ReferenceError, _>(
        "Could not mutable borrow reference.".to_owned(),
    ))
}

#[pyclass(name=StepInfo,subclass)]
#[derive(Clone)]
struct PyStepInfo {
    /// Should we put this into Arc ?
    inner: Arc<StepInfo>,
}

impl From<&StepInfo> for PyStepInfo {
    fn from(source: &StepInfo) -> Self {
        Self {
            inner: Arc::new(source.clone()),
        }
    }
}

#[pymethods]
impl PyStepInfo {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Default::default(),
        }
    }

    #[getter]
    fn get_current_latex(&self) -> PyResult<String> {
        Ok(self.inner.current_latex.clone())
    }

    #[setter]
    fn set_current_latex(&mut self, value: String) -> PyResult<()> {
        get_mut(&mut self.inner)?.current_latex = value;
        Ok(())
    }

    #[getter]
    fn get_value(&self) -> PyResult<Option<f32>> {
        Ok(self.inner.value)
    }

    #[setter]
    fn set_value(&mut self, value: f32) -> PyResult<()> {
        get_mut(&mut self.inner)?.value = Some(value);
        Ok(())
    }

    #[getter]
    fn get_confidence(&self) -> PyResult<Option<f32>> {
        Ok(self.inner.confidence)
    }

    #[setter]
    fn set_confidence(&mut self, confidence: f32) -> PyResult<()> {
        get_mut(&mut self.inner)?.confidence = Some(confidence);
        Ok(())
    }

    #[getter]
    fn get_subsequent(&self) -> PyResult<Vec<Self>> {
        Ok(self.inner.subsequent.iter().map(Self::from).collect())
    }

    fn add_subsequent(&mut self, other: Self) -> PyResult<()> {
        get_mut(&mut self.inner)?
            .subsequent
            .push((*other.inner).clone());
        Ok(())
    }

    #[getter]
    fn get_rule_id(&self) -> PyResult<u32> {
        Ok(self.inner.rule_id)
    }

    #[setter]
    fn set_rule_id(&mut self, value: u32) -> PyResult<()> {
        get_mut(&mut self.inner)?.rule_id = value;
        Ok(())
    }

    #[getter]
    fn get_path(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.path.clone())
    }

    #[setter]
    fn set_path(&mut self, path: Vec<usize>) -> PyResult<()> {
        get_mut(&mut self.inner)?.path = path;
        Ok(())
    }

    #[getter]
    fn get_top(&self) -> PyResult<u32> {
        Ok(self.inner.top)
    }

    #[setter]
    fn set_top(&mut self, value: u32) -> PyResult<()> {
        get_mut(&mut self.inner)?.top = value;
        Ok(())
    }

    #[getter]
    fn get_contributed(&self) -> PyResult<bool> {
        Ok(self.inner.contributed)
    }

    #[setter]
    fn set_contributed(&mut self, value: bool) -> PyResult<()> {
        get_mut(&mut self.inner)?.contributed = value;
        Ok(())
    }
}

#[pyproto]
impl PyNumberProtocol for PyStepInfo {
    fn __iadd__(&mut self, other: Self) {
        get_mut(&mut self.inner)
            .unwrap()
            .subsequent
            .push((*other.inner).clone());
    }
}

#[pyproto]
impl PyObjectProtocol for PyStepInfo {
    fn __richcmp__(&self, other: PyStepInfo, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }
}

#[pyclass(name=TraceStatistics,subclass)]
#[derive(Clone)]
struct PyTraceStatistics {
    inner: Arc<TraceStatistics>,
}

impl From<&TraceStatistics> for PyTraceStatistics {
    fn from(source: &TraceStatistics) -> Self {
        Self {
            inner: Arc::new((*source).clone()),
        }
    }
}

#[pymethods]
impl PyTraceStatistics {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Arc::new(Default::default()),
        }
    }

    #[getter]
    fn get_success(&self) -> PyResult<bool> {
        Ok(self.inner.success)
    }

    #[setter]
    fn set_success(&mut self, value: bool) -> PyResult<()> {
        get_mut(&mut self.inner)?.success = value;
        Ok(())
    }

    #[getter]
    fn get_fit_tries(&self) -> PyResult<u32> {
        Ok(self.inner.fit_tries)
    }

    #[setter]
    fn set_fit_tries(&mut self, value: u32) -> PyResult<()> {
        get_mut(&mut self.inner)?.fit_tries = value;
        Ok(())
    }

    #[getter]
    fn get_fit_results(&self) -> PyResult<u32> {
        Ok(self.inner.fit_results)
    }

    #[setter]
    fn set_fit_results(&mut self, value: u32) -> PyResult<()> {
        get_mut(&mut self.inner)?.fit_results = value;
        Ok(())
    }

    #[getter]
    fn get_trace(&self) -> PyResult<PyStepInfo> {
        Ok(PyStepInfo::from(&self.inner.trace))
    }

    #[setter]
    fn set_trace(&mut self, trace: PyStepInfo) -> PyResult<()> {
        get_mut(&mut self.inner)?.trace = (*trace.inner).clone();
        Ok(())
    }
}

#[pyclass(name=IterationSummary,subclass)]
#[derive(Clone)]
struct PyIterationSummary {
    inner: IterationSummary,
}

impl From<&IterationSummary> for PyIterationSummary {
    fn from(source: &IterationSummary) -> Self {
        Self {
            inner: (*source).clone(),
        }
    }
}

#[pymethods]
impl PyIterationSummary {
    #[new]
    fn py_new(
        fit_results: Option<u32>,
        success: Option<bool>,
        max_depth: Option<u32>,
        depth_of_solution: Option<u32>,
    ) -> Self {
        let fit_results = fit_results.unwrap_or_default();
        let success = success.unwrap_or_default();
        let max_depth = max_depth.unwrap_or_default();
        Self {
            inner: IterationSummary {
                fit_results,
                success,
                max_depth,
                depth_of_solution,
            },
        }
    }

    #[getter]
    fn get_fit_results(&self) -> PyResult<u32> {
        Ok(self.inner.fit_results)
    }

    #[setter]
    fn set_fit_results(&mut self, value: u32) -> PyResult<()> {
        self.inner.fit_results = value;
        Ok(())
    }

    #[getter]
    fn get_success(&self) -> PyResult<bool> {
        Ok(self.inner.success)
    }

    #[setter]
    fn set_success(&mut self, value: bool) -> PyResult<()> {
        self.inner.success = value;
        Ok(())
    }

    #[getter]
    fn get_max_depth(&self) -> PyResult<u32> {
        Ok(self.inner.max_depth)
    }

    #[getter]
    fn get_max_depth_of_solution(&self) -> PyResult<Option<u32>> {
        Ok(self.inner.depth_of_solution)
    }
}

#[pyproto]
impl PyObjectProtocol for PyIterationSummary {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __richcmp__(&self, other: PyIterationSummary, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }
}

#[pyclass(name=ProblemSummary,subclass)]
#[derive(Clone)]
struct PyProblemSummary {
    inner: ProblemSummary,
}

impl From<&ProblemSummary> for PyProblemSummary {
    fn from(source: &ProblemSummary) -> Self {
        Self {
            inner: (*source).clone(),
        }
    }
}

#[pyclass(name=ProblemStatistics,subclass)]
#[derive(Clone)]
struct PyProblemStatistics {
    inner: Arc<ProblemStatistics>,
}

impl From<&ProblemStatistics> for PyProblemStatistics {
    fn from(source: &ProblemStatistics) -> Self {
        Self {
            inner: Arc::new((*source).clone()),
        }
    }
}

#[pymethods]
impl PyProblemStatistics {
    #[new]
    fn py_new(problem_name: String) -> Self {
        Self {
            inner: Arc::new(ProblemStatistics {
                problem_name,
                iterations: Vec::new(),
            }),
        }
    }

    #[getter]
    fn get_problem_name(&self) -> PyResult<String> {
        Ok(self.inner.problem_name.clone())
    }

    #[setter]
    fn set_problem_name(&mut self, value: String) -> PyResult<()> {
        get_mut(&mut self.inner)?.problem_name = value;
        Ok(())
    }

    #[getter]
    fn get_iterations(&self) -> PyResult<Vec<PyTraceStatistics>> {
        Ok(self
            .inner
            .iterations
            .iter()
            .map(PyTraceStatistics::from)
            .collect())
    }

    fn add_iteration(&mut self, trace: &PyTraceStatistics) -> PyResult<()> {
        get_mut(&mut self.inner)?
            .iterations
            .push((*trace.inner).clone());
        Ok(())
    }
}

#[pyproto]
impl PyNumberProtocol for PyProblemStatistics {
    fn __iadd__(&mut self, trace: PyTraceStatistics) {
        get_mut(&mut self.inner)
            .unwrap()
            .iterations
            .push((*trace.inner).clone());
    }
}

#[pymethods]
impl PyProblemSummary {
    #[new]
    fn py_new(
        name: String,
        success: bool,
        iterations: Option<Vec<PyIterationSummary>>,
        initial_latex: Option<String>,
    ) -> Self {
        let iterations = iterations
            .unwrap_or(vec![])
            .into_iter()
            .map(|w| w.inner)
            .collect();
        Self {
            inner: ProblemSummary {
                name,
                success,
                iterations,
                initial_latex,
            },
        }
    }

    #[getter]
    fn get_initial_latex(&self) -> PyResult<Option<String>> {
        Ok(self.inner.initial_latex.clone())
    }

    #[getter]
    fn get_iterations(&self) -> PyResult<Vec<PyIterationSummary>> {
        Ok(self
            .inner
            .iterations
            .iter()
            .map(PyIterationSummary::from)
            .collect())
    }

    #[getter]
    fn get_name(&self) -> PyResult<String> {
        Ok(self.inner.name.clone())
    }

    #[getter]
    fn get_success(&self) -> PyResult<bool> {
        Ok(self.inner.success)
    }
}

#[pyproto]
impl PyObjectProtocol for PyProblemSummary {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __richcmp__(&self, other: PyProblemSummary, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }
}

#[pyclass(name=SolverStatistics,subclass)]
struct PySolverStatistics {
    inner: Arc<SolverStatistics>,
}

#[pymethods]
impl PySolverStatistics {
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Arc::new(Default::default()),
        }
    }

    fn get_problem(&self, problem_name: &str) -> PyResult<PyProblemStatistics> {
        let problem = self
            .inner
            .problems
            .get(problem_name)
            .ok_or(PyErr::new::<KeyError, _>(format!(
                "No problem found with name {}.",
                problem_name
            )))?;

        Ok(PyProblemStatistics::from(problem))
    }

    fn add_problem(&mut self, problem: PyProblemStatistics) -> PyResult<()> {
        get_mut(&mut self.inner)?
            .problems
            .insert(problem.inner.problem_name.clone(), (*problem.inner).clone());
        Ok(())
    }

    #[getter]
    fn header(&self) -> PyResult<Vec<PyProblemSummary>> {
        Ok(self
            .inner
            .summaries()
            .iter()
            .map(PyProblemSummary::from)
            .collect())
    }

    #[staticmethod]
    fn load(filename: &str) -> PyResult<Self> {
        let statistics = SolverStatistics::load(filename).map_err(|msg| {
            PyErr::new::<FileNotFoundError, _>(format!(
                "Could not load from file {}: {}",
                filename, msg
            ))
        })?;

        Ok(Self {
            inner: Arc::new(statistics),
        })
    }

    fn dump(&self, filename: &str) -> PyResult<()> {
        self.inner.dump(filename).map_err(|msg| {
            PyErr::new::<IOError, _>(format!("Could not dump to file {}: {}", filename, msg))
        })
    }
}

#[pyproto]
impl PyNumberProtocol for PySolverStatistics {
    fn __iadd__(&mut self, problem: PyProblemStatistics) {
        get_mut(&mut self.inner)
            .unwrap()
            .problems
            .insert(problem.inner.problem_name.clone(), (*problem.inner).clone());
    }
}

#[pyproto]
impl PyMappingProtocol for PySolverStatistics {
    fn __len__(&self) -> usize {
        self.inner.problems.len()
    }

    fn __getitem__(&self, query: String) -> PyResult<PyProblemStatistics> {
        let problem = self
            .inner
            .problems
            .get(&query)
            .ok_or(PyErr::new::<KeyError, _>(format!(
                "No problem found with name {}.",
                query
            )))?;

        Ok(PyProblemStatistics::from(problem))
    }
}

/// Registers all functions and classes regarding Solver trace.
pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyStepInfo>()?;
    m.add_class::<PyTraceStatistics>()?;
    m.add_class::<PyProblemStatistics>()?;
    m.add_class::<PySolverStatistics>()?;
    m.add_class::<PyProblemSummary>()?;
    m.add_class::<PyIterationSummary>()?;
    Ok(())
}
