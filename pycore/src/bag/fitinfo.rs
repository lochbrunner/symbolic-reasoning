use pyo3::class::basic::CompareOp;
use pyo3::class::PyObjectProtocol;
use pyo3::exceptions::NotImplementedError;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use core::bag;

use crate::common::op_to_string;

#[pyclass(name=FitInfo,subclass)]
#[derive(Clone)]
pub struct PyFitInfo {
    /// Starting with 1 for better embedding
    pub data: Arc<bag::FitInfo>,
}

impl PyFitInfo {
    pub fn new(orig: bag::FitInfo) -> PyFitInfo {
        PyFitInfo {
            data: Arc::new(orig),
        }
    }
}

#[pymethods]
impl PyFitInfo {
    #[new]
    fn py_new(rule_id: u32, path: Vec<usize>, positive: bool) -> Self {
        PyFitInfo::new(bag::FitInfo {
            rule_id,
            path,
            policy: bag::Policy::new(positive),
        })
    }

    #[getter]
    fn rule(&self) -> PyResult<u32> {
        Ok(self.data.rule_id)
    }

    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.data.path.clone())
    }

    #[getter]
    fn policy(&self) -> PyResult<f32> {
        Ok(self.data.policy.value())
    }
}

#[pyproto]
impl PyObjectProtocol for PyFitInfo {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.data))
    }

    fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(*self.data == *other.data),
            CompareOp::Ne => Ok(*self.data != *other.data),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut state = DefaultHasher::new();
        (*self.data).hash(&mut state);
        Ok(state.finish() as isize)
    }
}
