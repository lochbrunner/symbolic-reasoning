use pyo3::exceptions::IndexError;
use pyo3::prelude::*;

use crate::symbol::PySymbol;
use core::Symbol;

/// Client API to build a Symbol
/// Advantage over PySymbol is that it has exclusive ownership of Symbol
#[pyclass(name=SymbolBuilder,subclass)]
pub struct PySymbolBuilder {
    pub inner: Symbol,
}

fn create_empty_symbol() -> Symbol {
    Symbol {
        ident: "".to_string(),
        childs: Vec::new(),
        depth: 1,
        flags: 0,
        value: None,
    }
}

#[pymethods]
impl PySymbolBuilder {
    #[new]
    fn py_new() -> Self {
        PySymbolBuilder {
            inner: create_empty_symbol(),
        }
    }

    /// Adds childs at each arm uniformly
    #[args(child_per_arm = 2)]
    fn add_level_uniform(&mut self, child_per_arm: usize) -> PyResult<()> {
        let depth = self.inner.depth;
        for leaves in self.inner.iter_level_mut(depth - 1) {
            for _ in 0..child_per_arm {
                leaves.childs.push(create_empty_symbol());
            }
        }
        self.inner.fix_depth();
        Ok(())
    }

    /// Sets the idents of all symbols of the specified level in the order of traversing
    /// to the given ident.
    /// If there are more entries in the list the remaining get ignored.
    fn set_level_idents(&mut self, level: u32, idents: Vec<String>) -> PyResult<()> {
        for (i, node) in self.inner.iter_level_mut(level).enumerate() {
            let ident = match idents.get(i) {
                None => {
                    return Err(PyErr::new::<IndexError, _>(
                        "Idents list does not contain enough entries",
                    ))
                }
                Some(ident) => ident,
            };
            node.ident = ident.clone();
        }
        Ok(())
    }

    fn get_level_idents(&self, level: u32) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .iter_level(level)
            .map(|s| s.ident.clone())
            .collect())
    }

    #[getter]
    fn symbol(&self) -> PyResult<PySymbol> {
        Ok(PySymbol::new(self.inner.clone()))
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for PySymbolBuilder {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }
}
