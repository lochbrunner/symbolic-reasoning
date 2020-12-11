//! This module contains containers to store tracing results of the solver.
//! Especially for the t3_loop.py script.
//!
use crate::common::op_to_string;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{NotImplementedError, ReferenceError};
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
use std::collections::HashMap;
use std::sync::Arc;

use core::io::solver_trace::{
    ProblemStatistics, ProblemSummary, SolverStatistics, StepInfo, TraceStatistics,
};

// macro_rules! create_properties {
//     ($class:ty, {$($member_name:ident : $member_type:ty),*}) => {
//         #[pymethods]
//         impl $class {
//             $(
//                 paste! {
//                     #[getter]
//                     fn [<get_ $member_name>](&self) -> PyResult<$member_type> {
//                         Ok(self.inner.$member_name)
//                     }
//                     #[setter]
//                     fn [<set_ $member_name>](&mut self, value: $member_type) -> PyResult<()> {
//                         self.inner.$member_name = value;
//                         Ok(())
//                     }
//                 }
//             )*
//         }
//     };
// }

#[pyclass(name=StepInfo,subclass)]
#[derive(Clone)]
pub struct PyStepInfo {
    /// Should we put this into Arc ?
    inner: StepInfo,
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
        self.inner.current_latex = value;
        Ok(())
    }

    #[getter]
    fn get_value(&self) -> PyResult<Option<f32>> {
        Ok(self.inner.value)
    }

    #[setter]
    fn set_value(&mut self, value: f32) -> PyResult<()> {
        self.inner.value = Some(value);
        Ok(())
    }

    #[getter]
    fn get_confidence(&self) -> PyResult<Option<f32>> {
        Ok(self.inner.confidence)
    }

    #[setter]
    fn set_confidence(&mut self, confidence: f32) -> PyResult<()> {
        self.inner.confidence = Some(confidence);
        Ok(())
    }

    #[getter]
    fn get_subsequent(&self) -> PyResult<Vec<Self>> {
        Ok(self
            .inner
            .subsequent
            .iter()
            .cloned()
            .map(|inner| Self { inner })
            .collect())
    }

    fn add_subsequent(&mut self, other: Self) -> PyResult<()> {
        self.inner.subsequent.push(other.inner);
        Ok(())
    }

    #[getter]
    fn get_rule_id(&self) -> PyResult<u32> {
        Ok(self.inner.rule_id)
    }

    #[setter]
    fn set_rule_id(&mut self, value: u32) -> PyResult<()> {
        self.inner.rule_id = value;
        Ok(())
    }

    #[getter]
    fn get_path(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.path.clone())
    }

    #[setter]
    fn set_path(&mut self, path: Vec<usize>) -> PyResult<()> {
        self.inner.path = path;
        Ok(())
    }

    #[getter]
    fn get_top(&self) -> PyResult<u32> {
        Ok(self.inner.top)
    }

    #[setter]
    fn set_top(&mut self, value: u32) -> PyResult<()> {
        self.inner.top = value;
        Ok(())
    }

    #[getter]
    fn get_contributed(&self) -> PyResult<bool> {
        Ok(self.inner.contributed)
    }

    #[setter]
    fn set_contributed(&mut self, value: bool) -> PyResult<()> {
        self.inner.contributed = value;
        Ok(())
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
pub struct PyTraceStatistics {
    inner: Arc<TraceStatistics>,
}

#[pymethods]
impl PyTraceStatistics {
    #[getter]
    fn get_success(&self) -> PyResult<bool> {
        Ok(self.inner.success)
    }

    #[setter]
    fn set_success(&mut self, value: bool) -> PyResult<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<ReferenceError, _>(
            "Could not mutable borrow reference.".to_owned(),
        ))?;
        inner.success = value;
        Ok(())
    }

    #[getter]
    fn get_fit_tries(&self) -> PyResult<u32> {
        Ok(self.inner.fit_tries)
    }

    #[setter]
    fn set_fit_tries(&mut self, value: u32) -> PyResult<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<ReferenceError, _>(
            "Could not mutable borrow reference.".to_owned(),
        ))?;
        inner.fit_tries = value;
        Ok(())
    }

    #[getter]
    fn get_fit_results(&self) -> PyResult<u32> {
        Ok(self.inner.fit_results)
    }

    #[setter]
    fn set_fit_results(&mut self, value: u32) -> PyResult<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<ReferenceError, _>(
            "Could not mutable borrow reference.".to_owned(),
        ))?;
        inner.fit_results = value;
        Ok(())
    }

    #[getter]
    fn get_trace(&self) -> PyResult<PyStepInfo> {
        Ok(PyStepInfo {
            inner: self.inner.trace.clone(),
        })
    }

    #[setter]
    fn set_trace(&mut self, trace: PyStepInfo) -> PyResult<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<ReferenceError, _>(
            "Could not mutable borrow reference.".to_owned(),
        ))?;
        inner.trace = trace.inner;
        Ok(())
    }
}

#[pyclass(name=ProblemSummary,subclass)]
pub struct PyProblemSummary {
    problem_name: String,
    success: bool,
}

impl From<&ProblemSummary> for PyProblemSummary {
    fn from(source: &ProblemSummary) -> Self {
        Self {
            problem_name: source.problem_name.clone(),
            success: source.success,
        }
    }
}

#[pyclass(name=ProblemStatistics,subclass)]
struct PyProblemStatistics {
    inner: Arc<ProblemStatistics>,
}

#[pymethods]
impl PyProblemStatistics {
    #[getter]
    fn get_problem_name(&self) -> PyResult<String> {
        Ok(self.inner.problem_name.clone())
    }

    #[setter]
    fn set_problem_name(&mut self, value: String) -> PyResult<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<ReferenceError, _>(
            "Could not mutable borrow reference.".to_owned(),
        ))?;
        inner.problem_name = value;
        Ok(())
    }
}

#[pymethods]
impl PyProblemSummary {
    #[getter]
    fn get_problem_name(&self) -> PyResult<String> {
        Ok(self.problem_name.clone())
    }

    #[getter]
    fn get_success(&self) -> PyResult<bool> {
        Ok(self.success)
    }
}

#[pyclass(name=SolverStatistics,subclass)]
pub struct PySolverStatistics {
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

    // #[getter]
    // fn header(&self) -> PyResult<Vec<PyProblemSummary>> {
    //     self.problems
    //         .values()
    //         .map(|p| p.summary())
    //         .collect::<PyResult<_>>()
    // }
}
