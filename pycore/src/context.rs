use core::Declaration;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python Wrapper for core::Declaration
#[pyclass(name=Declaration,subclass)]
pub struct PyDeclaration {
    pub inner: Declaration,
}

#[pymethods]
impl PyDeclaration {
    #[new]
    fn py_new() -> Self {
        PyDeclaration {
            inner: Declaration {
                is_fixed: false,
                is_function: false,
                only_root: false,
            },
        }
    }

    #[getter]
    fn get_is_fixed(&self) -> PyResult<bool> {
        Ok(self.inner.is_fixed)
    }

    #[setter]
    fn set_is_fixed(&mut self, is_fixed: bool) -> PyResult<()> {
        self.inner.is_fixed = is_fixed;
        Ok(())
    }

    #[getter]
    fn get_is_function(&self) -> PyResult<bool> {
        Ok(self.inner.is_function)
    }

    #[setter]
    fn set_is_function(&mut self, is_function: bool) -> PyResult<()> {
        self.inner.is_function = is_function;
        Ok(())
    }

    #[getter]
    fn get_only_root(&self) -> PyResult<bool> {
        Ok(self.inner.only_root)
    }

    #[setter]
    fn set_only_root(&mut self, only_root: bool) -> PyResult<()> {
        self.inner.only_root = only_root;
        Ok(())
    }
}

#[pyproto]
impl PyObjectProtocol for PyDeclaration {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
}

/// Python Wrapper for core::Context
#[pyclass(name=Context,subclass)]
pub struct PyContext {
    pub inner: core::Context,
}

#[pymethods]
impl PyContext {
    #[new]
    fn py_new() -> Self {
        PyContext {
            inner: {
                core::Context {
                    declarations: HashMap::new(),
                }
            },
        }
    }

    #[staticmethod]
    fn standard() -> PyResult<PyContext> {
        Ok(PyContext {
            inner: core::Context::standard(),
        })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<PyContext> {
        match core::Context::load(&path) {
            Ok(inner) => Ok(PyContext { inner }),
            Err(msg) => Err(PyErr::new::<exceptions::TypeError, _>(msg)),
        }
    }

    fn add_function(&mut self, name: String, fixed: Option<bool>) -> PyResult<()> {
        self.inner
            .declarations
            .insert(name, Declaration::function(fixed.unwrap_or(false)));
        Ok(())
    }

    fn add_constant(&mut self, name: String) -> PyResult<()> {
        self.inner
            .declarations
            .insert(name, Declaration::constant());
        Ok(())
    }

    #[getter]
    fn declarations(&self) -> PyResult<HashMap<String, PyDeclaration>> {
        Ok(self
            .inner
            .declarations
            .iter()
            .map(|(name, dec)| (name.clone(), PyDeclaration { inner: dec.clone() }))
            .collect())
    }
}

#[pyproto]
impl PyObjectProtocol for PyContext {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
}
