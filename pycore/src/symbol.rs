use crate::context::PyContext;
use core::dumper::{dump_latex, dump_verbose};
use core::Symbol;
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::{IndexError, TypeError};
use pyo3::prelude::*;
use std::fmt;
use std::rc::Rc;

/// Python Wrapper for core::Symbol
#[pyclass(name=Symbol,subclass)]
#[derive(Hash, Clone, PartialEq, Eq)]
pub struct PySymbol {
    pub inner: Rc<Symbol>,
}

impl PySymbol {
    pub fn new(symbol: Symbol) -> PySymbol {
        PySymbol {
            inner: Rc::new(symbol),
        }
    }
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
                Ok(Some(PySymbol::new(current)))
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
                Ok(Some((path, PySymbol::new(symbol))))
            }
        }
    }
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    fn parse(context: &PyContext, code: String) -> PyResult<PySymbol> {
        let inner =
            Symbol::parse(&context.inner, &code).map_err(|msg| PyErr::new::<TypeError, _>(msg))?;
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn ident(&self) -> PyResult<String> {
        Ok(self.inner.ident.clone())
    }

    #[text_signature = "($self, path, /)"]
    fn at(&self, path: Vec<usize>) -> PyResult<PySymbol> {
        match self.inner.at(&path) {
            None => Err(PyErr::new::<IndexError, _>("Index is out of bound")),
            Some(item) => Ok(PySymbol::new(item.clone())),
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

    /// LaTeX representation of that node
    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(dump_latex(&self.inner, None))
    }

    /// The node as a tree
    #[getter]
    fn tree(&self) -> PyResult<String> {
        Ok(self.inner.print_tree())
    }

    #[getter]
    fn childs(&self) -> PyResult<Vec<PySymbol>> {
        Ok(self
            .inner
            .childs
            .iter()
            .map(|s| PySymbol::new(s.clone()))
            .collect())
    }

    #[text_signature = "($self, /)"]
    fn clone(&self) -> PyResult<PySymbol> {
        Ok(PySymbol {
            inner: self.inner.clone(),
        })
    }

    #[text_signature = "($self, padding, spread, depth, /)"]
    fn pad(&mut self, padding: String, spread: u32, depth: u32) -> PyResult<()> {
        match Rc::get_mut(&mut self.inner) {
            None => {
                return Err(PyErr::new::<TypeError, _>(format!(
                    "Can not get mut reference of symbol {}",
                    self.inner
                )));
            }
            Some(parent) => {
                for level in 0..depth {
                    for node in parent.iter_level_mut(level) {
                        let nc = node.childs.len() as u32;
                        for _ in nc..spread {
                            node.childs.push(Symbol::new_variable(&padding, false));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    #[text_signature = "($self, padding, spread, depth, /)"]
    fn create_padded(&self, padding: String, spread: u32, depth: u32) -> PyResult<PySymbol> {
        let mut new_symbol = (*self.inner).clone();
        for level in 0..depth {
            for node in new_symbol.iter_level_mut(level) {
                let nc = node.childs.len() as u32;
                for _ in nc..spread {
                    node.childs.push(Symbol::new_variable(&padding, false));
                }
            }
        }
        Ok(PySymbol::new(new_symbol))
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

impl fmt::Debug for PySymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
