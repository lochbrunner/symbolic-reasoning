use crate::context::PyContext;
use crate::symbol::PySymbol;
use core::dumper::{dump_latex, dump_symbol_plain};
use core::Rule;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::convert::From;
use std::sync::Arc;

/// Python Wrapper for core::Rule
#[pyclass(name=Rule,subclass)]
#[derive(Clone)]
pub struct PyRule {
    pub inner: Arc<Rule>,
}

impl From<&Rule> for PyRule {
    fn from(rule: &Rule) -> Self {
        Self {
            inner: Arc::new(rule.clone()),
        }
    }
}

#[pymethods]
impl PyRule {
    #[new]
    fn py_new(condition: PySymbol, conclusion: PySymbol, name: &str) -> Self {
        PyRule {
            inner: Arc::new(Rule {
                condition: (*condition.inner).clone(),
                conclusion: (*conclusion.inner).clone(),
                name: name.to_owned(),
            }),
        }
    }

    /// Just uses the first rule.
    #[staticmethod]
    fn parse(context: &PyContext, code: String, name: Option<String>) -> PyResult<PyRule> {
        match Rule::parse(&context.inner, &code) {
            Ok(mut rule) => {
                if let Some(mut rule) = rule.pop() {
                    rule.name = name.unwrap_or(format!("Parsed from {}", code));
                    Ok(PyRule {
                        inner: Arc::new(rule),
                    })
                } else {
                    Err(PyErr::new::<exceptions::ValueError, _>(format!(
                        "Can not parse rule from {}",
                        code
                    )))
                }
            }
            Err(msg) => Err(PyErr::new::<exceptions::TypeError, _>(msg)),
        }
    }

    #[getter]
    fn get_condition(&self) -> PyResult<PySymbol> {
        let inner = self.inner.condition.clone();
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn get_conclusion(&self) -> PyResult<PySymbol> {
        let inner = self.inner.conclusion.clone();
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn reverse(&self) -> PyResult<PyRule> {
        Ok(PyRule {
            inner: Arc::new(Rule {
                conclusion: self.inner.condition.clone(),
                condition: self.inner.conclusion.clone(),
                name: format!("Reverse of {}", &self.inner.name),
            }),
        })
    }

    /// Dumps the verbose order of operators with equal precedence
    #[getter]
    fn verbose(&self) -> PyResult<String> {
        Ok(format!(
            "{} => {} ",
            dump_symbol_plain(&self.inner.condition, true),
            dump_symbol_plain(&self.inner.conclusion, true)
        ))
    }

    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(format!(
            "{} \\Rightarrow {} ",
            dump_latex(&self.inner.condition, vec![], false),
            dump_latex(&self.inner.conclusion, vec![], false)
        ))
    }

    #[getter]
    fn latex_verbose(&self) -> PyResult<String> {
        Ok(format!(
            "{} \\Rightarrow {} ",
            dump_latex(&self.inner.condition, vec![], true),
            dump_latex(&self.inner.conclusion, vec![], true)
        ))
    }

    #[getter]
    fn get_name(&self) -> PyResult<String> {
        Ok(self.inner.name.clone())
    }

    #[setter]
    fn set_name(&mut self, name: &str) -> PyResult<()> {
        if let Some(rule) = Arc::get_mut(&mut self.inner) {
            rule.name = name.to_owned();
            Ok(())
        } else {
            Err(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference of rule".to_owned(),
            ))
        }
    }
}

#[pyproto]
impl PyObjectProtocol for PyRule {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
