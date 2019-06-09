use crate::context::PyContext;
use core::Symbol;
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::IndexError;
use pyo3::prelude::*;
use std::rc::Rc;

/// Python Wrapper for core::Symbol
#[pyclass(name=Symbol,subclass)]
pub struct PySymbol {
    pub inner: Rc<Symbol>,
}

#[pyclass(name=SymbolIter)]
#[derive(Clone)]
pub struct PySymbolIter {
    pub parent: Rc<Symbol>,
    pub stack: Vec<Symbol>,
}

#[pyproto]
impl PyIterProtocol for PySymbolIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolIter> {
        Ok(PySymbolIter {
            parent: s.parent.clone(),
            stack: vec![(*s.parent).clone()],
        })
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<PySymbol>> {
        match s.stack.pop() {
            None => Ok(None),
            Some(current) => {
                for child in current.childs.iter() {
                    s.stack.push(child.clone());
                }
                Ok(Some(PySymbol {
                    inner: Rc::new(current),
                }))
            }
        }
    }
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PySymbol> {
        let inner = Symbol::parse_from_str(&context.inner, code);
        Ok(PySymbol {
            inner: Rc::new(inner),
        })
    }

    #[getter]
    fn ident(&self) -> PyResult<String> {
        Ok(self.inner.ident.clone())
    }

    fn get(&self, path: Vec<usize>) -> PyResult<PySymbol> {
        match self.inner.get(&path) {
            None => Err(PyErr::new::<IndexError, _>("Index is out of bound")),
            Some(item) => Ok(PySymbol {
                inner: Rc::new(item.clone()),
            }),
        }
    }

    #[getter]
    fn parts(&self) -> PyResult<PySymbolIter> {
        Ok(PySymbolIter {
            parent: self.inner.clone(),
            stack: vec![(*self.inner).clone()],
        })
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for PySymbol {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
