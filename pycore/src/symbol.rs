use crate::context::PyContext;
use core::Symbol;
use pyo3::prelude::*;

/// Python Wrapper for core::Symbol
#[pyclass(name=Symbol,subclass)]
pub struct PySymbol {
    pub inner: Symbol,
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PySymbol> {
        let inner = Symbol::parse_from_str(&context.inner, code);
        Ok(PySymbol { inner })
    }

    fn ident(&self) -> PyResult<String> {
        Ok(self.inner.ident.clone())
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for PySymbol {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
