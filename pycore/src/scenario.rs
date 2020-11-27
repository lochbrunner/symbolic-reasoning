use core::scenario::{Scenario, ScenarioProblems};
use core::Rule;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::context::PyContext;
use crate::rule::PyRule;

#[pyclass(name=ScenarioProblems,subclass)]
pub struct PyScenarioProblems {
    pub inner: Arc<ScenarioProblems>,
}

fn export_rules<'a, I>(rules: I) -> HashMap<String, PyRule>
where
    I: Iterator<Item = (&'a String, &'a Rule)>,
{
    rules
        .map(|(k, v)| {
            (
                k.clone(),
                PyRule {
                    inner: Arc::new(v.clone()),
                    name: k.clone(),
                },
            )
        })
        .collect()
}

#[pymethods]
impl PyScenarioProblems {
    #[getter]
    fn validation(&self) -> PyResult<HashMap<String, PyRule>> {
        Ok(export_rules(self.inner.validation.iter()))
    }

    #[getter]
    fn training(&self) -> PyResult<HashMap<String, PyRule>> {
        Ok(export_rules(self.inner.training.iter()))
    }

    #[getter]
    fn all(&self) -> PyResult<HashMap<String, PyRule>> {
        Ok(export_rules(
            self.inner
                .training
                .iter()
                .chain(self.inner.validation.iter()),
        ))
    }
}

/// Python Wrapper for core::io::Scenario
#[pyclass(name=Scenario,subclass)]
#[derive(Clone)]
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
    fn problems(&self) -> PyResult<PyScenarioProblems> {
        Ok(PyScenarioProblems {
            inner: Arc::new(self.inner.problems.clone()),
        })
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
