use crate::context::PyContext;
use crate::symbol::PySymbol;
use core::dumper::{dump_latex, dump_verbose};
use core::Rule;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::rc::Rc;

/// Python Wrapper for core::Rule
#[pyclass(name=Rule,subclass)]
pub struct PyRule {
    pub inner: Rc<Rule>,
    pub name: String,
}

#[pymethods]
impl PyRule {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PyRule> {
        match Rule::parse(&context.inner, &code) {
            Ok(mut rule) => Ok(PyRule {
                inner: Rc::new(rule.pop().unwrap()),
                name: format!("Parsed from {}", code),
            }),
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
            inner: Rc::new(Rule {
                conclusion: self.inner.condition.clone(),
                condition: self.inner.conclusion.clone(),
            }),
            name: format!("Reverse of {}", self.name),
        })
    }

    /// Dumps the verbose order of operators with equal precedence
    #[getter]
    fn verbose(&self) -> PyResult<String> {
        Ok(format!(
            "{} => {} ",
            dump_verbose(&self.inner.condition),
            dump_verbose(&self.inner.conclusion)
        ))
    }

    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(format!(
            "{} \\Rightarrow {} ",
            dump_latex(&self.inner.condition, None),
            dump_latex(&self.inner.conclusion, None)
        ))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        Ok(self.name.clone())
    }
}

#[pyproto]
impl PyObjectProtocol for PyRule {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
