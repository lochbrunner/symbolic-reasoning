use crate::fit::PyFitMap;
use crate::symbol::PySymbol;
use core::apply;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// TODO: Expose fit and apply as one "atomic" call to python

pub fn pyapply_impl(
    py: Python,
    mapping: &PyFitMap,
    variable_creator: PyObject,
    prev: &PySymbol,
    conclusion: &PySymbol,
) -> PyResult<PySymbol> {
    let mapping = mapping.get_raw();
    let applied = apply::<_, PyErr>(
        &mapping,
        || {
            let obj = variable_creator.call(py, PyTuple::empty(py), None)?;
            let symbol: &PySymbol = obj.extract(py)?;
            Ok((*symbol.inner).clone())
        },
        &prev.inner,
        &conclusion.inner,
    )?;
    Ok(PySymbol::new(applied))
}
