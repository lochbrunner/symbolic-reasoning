use crate::context::PyContext;
use crate::symbol::PySymbol;
use std::rc::Rc;
use core::Rule;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::prelude::*;

/// Python Wrapper for core::Rule
#[pyclass(name=Rule,subclass)]
pub struct PyRule {
    pub inner: Rule,
}

#[pymethods]
impl PyRule {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PyRule> {
        let inner = Rule::parse(&context.inner, &code);
        Ok(PyRule { inner })
    }

    #[getter]
    fn get_condition(&self) -> PyResult<PySymbol> {
        let inner = self.inner.condition.clone();
        Ok(PySymbol { inner: Rc::new(inner) })
    }

    #[getter]
    fn get_conclusion(&self) -> PyResult<PySymbol> {
        let inner = self.inner.conclusion.clone();
        Ok(PySymbol { inner: Rc::new(inner) })
    }

    #[getter]
    fn reverse(&self) -> PyResult<PyRule> {
        Ok(PyRule{
            inner: {
                Rule{
                    conclusion: self.inner.condition.clone(),
                    condition: self.inner.conclusion.clone(),
                }
            }
        })
    }
}

#[pyproto]
impl PyObjectProtocol for PyRule {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
