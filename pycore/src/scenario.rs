use core::scenario::Scenario;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

use crate::context::PyContext;
use crate::rule::PyRule;

/// Python Wrapper for core::io::Scenario
#[pyclass(name=Scenario,subclass)]
pub struct PyScenario {
    pub inner: Rc<Scenario>,
}

#[pymethods]
impl PyScenario {
    #[staticmethod]
    fn load(code: String) -> PyResult<PyScenario> {
        match Scenario::load_from_yaml(&code) {
            Ok(scenario) => Ok(PyScenario {
                inner: Rc::new(scenario),
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
                        inner: Rc::new(v.clone()),
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
                        inner: Rc::new(v.clone()),
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
}
