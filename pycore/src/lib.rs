use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod apply;
mod bag;
mod context;
mod fit;
mod rule;
mod scenario;
mod symbol;
mod symbol_builder;
mod trace;

/// Tries to fit the inner into the outer symbol.unimplemented!
/// Returns a list of possible mappings.
#[pyfunction]
#[text_signature = "(outer, inner, /)"]
fn fit(outer: &symbol::PySymbol, inner: &symbol::PySymbol) -> PyResult<Vec<fit::PyFitMap>> {
    fit::pyfit_impl(outer, inner)
}

#[pyfunction]
#[text_signature = "(mapping, variable_creator, prev, conclusion, /)"]
fn apply(
    py: Python,
    mapping: &fit::PyFitMap,
    variable_creator: PyObject,
    prev: &symbol::PySymbol,
    conclusion: &symbol::PySymbol,
) -> PyResult<symbol::PySymbol> {
    apply::pyapply_impl(py, mapping, variable_creator, prev, conclusion)
}

#[pymodule]
fn pycore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<context::PyDeclaration>()?;
    m.add_class::<context::PyContext>()?;
    m.add_class::<symbol::PySymbol>()?;
    m.add_class::<symbol_builder::PySymbolBuilder>()?;
    m.add_class::<rule::PyRule>()?;
    m.add_class::<trace::PyApplyInfo>()?;
    m.add_class::<trace::PyTraceStep>()?;
    m.add_class::<trace::PyTrace>()?;
    m.add_class::<bag::PyBag>()?;
    m.add_class::<bag::PyBagMeta>()?;
    m.add_class::<bag::PyFitInfo>()?;
    m.add_class::<bag::PyContainer>()?;
    m.add_class::<bag::PySample>()?;
    m.add_class::<scenario::PyScenario>()?;
    m.add_class::<fit::PyFitMap>()?;
    m.add_wrapped(wrap_pyfunction!(fit))?;
    m.add_wrapped(wrap_pyfunction!(apply))?;
    Ok(())
}
