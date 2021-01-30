use core::scenario::{Scenario, ScenarioProblems};
use core::Rule;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::iter::FromIterator;
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
    #[new]
    fn py_new() -> Self {
        Self {
            inner: Arc::new(Default::default()),
        }
    }

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

    #[getter]
    fn additional_idents(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.additional_idents.clone())
    }

    #[staticmethod]
    #[text_signature = "(filename, /)"]
    fn load(filename: &str) -> PyResult<Self> {
        let data =
            ScenarioProblems::load(filename).map_err(PyErr::new::<exceptions::IOError, _>)?;
        Ok(Self {
            inner: Arc::new(data),
        })
    }

    fn dump(&self, filename: &str) -> PyResult<()> {
        self.inner
            .dump(filename)
            .map_err(PyErr::new::<exceptions::IOError, _>)?;
        Ok(())
    }

    fn add_to_training(&mut self, rule: PyRule, name: &str) -> PyResult<()> {
        let problems =
            Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference.".to_owned(),
            ))?;

        problems
            .training
            .insert(name.to_owned(), (*rule.inner).clone());
        Ok(())
    }

    fn add_to_validation(&mut self, rule: PyRule, name: &str) -> PyResult<()> {
        let problems =
            Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference.".to_owned(),
            ))?;

        problems
            .validation
            .insert(name.to_owned(), (*rule.inner).clone());
        Ok(())
    }

    fn add_additional_idents(&mut self, idents: Vec<String>) -> PyResult<()> {
        let problems =
            Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference.".to_owned(),
            ))?;
        let unique_idents: HashSet<String> = HashSet::from_iter(
            idents
                .into_iter()
                .chain(problems.additional_idents.iter().cloned()),
        );
        problems.additional_idents.clear();
        problems.additional_idents.extend(unique_idents.into_iter());
        Ok(())
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
        let scenario =
            Scenario::load_from_yaml(&filename).map_err(PyErr::new::<exceptions::IOError, _>)?;
        Ok(Self {
            inner: Arc::new(scenario),
        })
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
