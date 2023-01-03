use crate::fit::PyFitMap;
use crate::symbol::PySymbol;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;

#[pyfunction]
#[text_signature = "(mapping, variable_creator, orig, conclusion, /)"]
fn apply(
    py: Python,
    mapping: &PyFitMap,
    variable_creator: PyObject,
    orig: &PySymbol,
    conclusion: &PySymbol,
) -> PyResult<PySymbol> {
    let mapping = mapping.get_raw();
    let applied = core::apply::<_, PyErr>(
        &mapping,
        || {
            let obj = variable_creator.call(py, PyTuple::empty(py), None)?;
            let symbol: PySymbol = obj.extract(py)?;
            Ok((*symbol.inner).clone())
        },
        &orig.inner,
        &conclusion.inner,
    )?;
    Ok(PySymbol::new(applied))
}

// TODO: apply_batch

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(apply))?;
    Ok(())
}
