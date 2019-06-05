use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Python Wrapper for core::Declaration
#[pyclass(name=Declaration,subclass)]
pub struct PyDeclaration {
    pub is_fixed: bool,
    pub is_function: bool,
    pub only_root: bool,
}

#[pymethods]
impl PyDeclaration {
    #[new]
    fn py_new(obj: &PyRawObject) {
        obj.init({
            PyDeclaration {
                is_fixed: false,
                is_function: false,
                only_root: false,
            }
        });
    }

    #[getter]
    fn get_is_fixed(&self) -> PyResult<bool> {
        Ok(self.is_fixed)
    }

    #[setter]
    fn set_is_fixed(&mut self, is_fixed: bool) -> PyResult<()> {
        self.is_fixed = is_fixed;
        Ok(())
    }

    #[getter]
    fn get_is_function(&self) -> PyResult<bool> {
        Ok(self.is_function)
    }

    #[setter]
    fn set_is_function(&mut self, is_function: bool) -> PyResult<()> {
        self.is_function = is_function;
        Ok(())
    }

    #[getter]
    fn get_only_root(&self) -> PyResult<bool> {
        Ok(self.only_root)
    }

    #[setter]
    fn set_only_root(&mut self, only_root: bool) -> PyResult<()> {
        self.only_root = only_root;
        Ok(())
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
    fn py_new(obj: &PyRawObject) {
        obj.init({
            PyContext {
                inner: {
                    core::Context {
                        declarations: HashMap::new(),
                    }
                },
            }
        });
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
}
