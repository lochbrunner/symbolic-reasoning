use core::scenario::{Scenario, ScenarioProblems};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;

use crate::context::PyContext;
use crate::rule::PyRule;

#[pyclass(name=ScenarioProblems,subclass)]
pub struct PyScenarioProblems {
    pub inner: Arc<ScenarioProblems>,
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
    fn validation(&self) -> PyResult<Vec<PyRule>> {
        Ok(self.inner.validation.iter().map(PyRule::from).collect())
    }

    #[getter]
    fn training(&self) -> PyResult<Vec<PyRule>> {
        Ok(self.inner.training.iter().map(PyRule::from).collect())
    }

    #[getter]
    fn all(&self) -> PyResult<Vec<PyRule>> {
        Ok(self
            .inner
            .training
            .iter()
            .chain(self.inner.validation.iter())
            .map(PyRule::from)
            .collect())
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

    fn add_to_training(&mut self, rule: PyRule) -> PyResult<()> {
        let problems =
            Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference.".to_owned(),
            ))?;

        problems.training.push((*rule.inner).clone());
        Ok(())
    }

    fn add_to_validation(&mut self, rule: PyRule) -> PyResult<()> {
        let problems =
            Arc::get_mut(&mut self.inner).ok_or(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference.".to_owned(),
            ))?;

        problems.validation.push((*rule.inner).clone());
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
    #[args(filename, no_dependencies = false)]
    fn load(filename: String, no_dependencies: bool) -> PyResult<PyScenario> {
        let scenario = Scenario::load_from_yaml(&filename, no_dependencies)
            .map_err(PyErr::new::<exceptions::IOError, _>)?;
        Ok(Self {
            inner: Arc::new(scenario),
        })
    }

    #[getter]
    fn rules(&self) -> PyResult<Vec<PyRule>> {
        Ok(self.inner.rules.iter().map(PyRule::from).collect())
    }

    #[getter]
    fn problems(&self) -> PyResult<Option<PyScenarioProblems>> {
        if let Some(ref problems) = self.inner.problems {
            Ok(Some(PyScenarioProblems {
                inner: Arc::new(problems.clone()),
            }))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn declarations(&self) -> PyResult<PyContext> {
        Ok(PyContext {
            inner: self.inner.declarations.clone(),
        })
    }

    #[args(ignore_declaration = true)]
    fn idents(&self, ignore_declaration: bool) -> PyResult<Vec<String>> {
        Ok(self.inner.idents(ignore_declaration))
    }

    /// With padding
    #[getter]
    fn tagset_size(&self) -> PyResult<usize> {
        Ok(self.inner.rules.len() + 1)
    }

    #[args(ignore_declaration = true)]
    fn vocab_size(&self, ignore_declaration: bool) -> PyResult<usize> {
        Ok(self.inner.idents(ignore_declaration).len())
    }

    #[getter]
    fn spread(&self) -> PyResult<u32> {
        Ok(2)
    }
}
