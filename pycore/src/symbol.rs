use crate::context::PyContext;
use core::dumper::{dump_latex, dump_verbose};
use core::Symbol;
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::{IndexError, TypeError};
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

#[pyclass(name=SymbolAndPathIter)]
#[derive(Clone)]
pub struct PySymbolAndPathIter {
    pub parent: Rc<Symbol>,
    pub stack: Vec<Vec<usize>>,
}

#[pyproto]
impl PyIterProtocol for PySymbolAndPathIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolAndPathIter> {
        Ok(PySymbolAndPathIter {
            parent: s.parent.clone(),
            stack: vec![vec![]],
        })
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<(Vec<usize>, PySymbol)>> {
        match s.stack.pop() {
            None => Ok(None),
            Some(path) => {
                let symbol = s
                    .parent
                    .at(&path)
                    .expect(&format!("part at path: {:?}", path))
                    .clone();
                for (i, _) in symbol.childs.iter().enumerate() {
                    s.stack.push([&path[..], &[i]].concat());
                }
                Ok(Some((
                    path,
                    PySymbol {
                        inner: Rc::new(symbol),
                    },
                )))
            }
        }
    }
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PySymbol> {
        match Symbol::parse(&context.inner, &code) {
            Ok(inner) => Ok(PySymbol {
                inner: Rc::new(inner),
            }),
            Err(msg) => Err(PyErr::new::<TypeError, _>(msg)),
        }
    }

    #[getter]
    fn ident(&self) -> PyResult<String> {
        Ok(self.inner.ident.clone())
    }

    fn at(&self, path: Vec<usize>) -> PyResult<PySymbol> {
        match self.inner.at(&path) {
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

    #[getter]
    fn parts_with_path(&self) -> PyResult<PySymbolAndPathIter> {
        Ok(PySymbolAndPathIter {
            parent: self.inner.clone(),
            stack: vec![vec![]],
        })
    }

    /// Dumps the verbose order of operators with equal precedence
    #[getter]
    fn verbose(&self) -> PyResult<String> {
        Ok(dump_verbose(&self.inner))
    }

    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(dump_latex(&self.inner, None))
    }

    #[getter]
    fn childs(&self) -> PyResult<Vec<PySymbol>> {
        Ok(self
            .inner
            .childs
            .iter()
            .map(|s| PySymbol {
                inner: Rc::new(s.clone()),
            })
            .collect())
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for PySymbol {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
}
