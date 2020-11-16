use core::scenario::Scenario;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::context::PyContext;
use crate::rule::PyRule;

/// Python Wrapper for core::io::Scenario
#[pyclass(name=Scenario,subclass)]
pub struct PyScenario {
    pub inner: Arc<Scenario>,
}

#[pymethods]
impl PyScenario {
    /// Loads a scenario from file
    #[staticmethod]
    #[text_signature = "(filename, /)"]
    fn load(filename: String) -> PyResult<PyScenario> {
        match Scenario::load_from_yaml(&filename) {
            Ok(scenario) => Ok(PyScenario {
                inner: Arc::new(scenario),
            }),
            Err(msg) => Err(PyErr::new::<exceptions::IOError, _>(msg)),
        }
    }

    #[getter]
    fn rules(&self) -> PyResult<HashMap<String, PyRule>> {
        Ok(self
            .inner
            .rules
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    PyRule {
                        inner: Arc::new(v.clone()),
                        name: k.clone(),
                    },
                )
            })
            .collect())
    }

    #[getter]
    fn problems(&self) -> PyResult<HashMap<String, PyRule>> {
        Ok(self
            .inner
            .problems
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    PyRule {
                        inner: Arc::new(v.clone()),
                        name: k.clone(),
                    },
                )
            })
            .collect())
    }

    #[getter]
    fn declarations(&self) -> PyResult<PyContext> {
        Ok(PyContext {
            inner: self.inner.declarations.clone(),
        })
    }

    #[getter]
    fn idents(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.idents())
    }

    /// With padding
    #[getter]
    fn tagset_size(&self) -> PyResult<usize> {
        Ok(self.inner.rules.len() + 1)
    }

    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        Ok(self.inner.idents().len())
    }

    #[getter]
    fn spread(&self) -> PyResult<u32> {
        Ok(2)
    }
}
