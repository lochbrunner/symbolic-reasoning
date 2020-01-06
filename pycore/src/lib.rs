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

/// Tries to fit the inner into the outer symbol.
/// Returns a list of possible mappings.
#[pyfunction]
#[text_signature = "(outer, inner, /)"]
fn fit(outer: &symbol::PySymbol, inner: &symbol::PySymbol) -> PyResult<Vec<fit::PyFitMap>> {
    fit::pyfit_impl(outer, inner)
}

/// Tries to fit the inner into the outer symbol at a specified path.
/// Returns a list of possible mappings.
#[pyfunction]
#[text_signature = "(outer, inner, path /)"]
fn fit_at(
    outer: &symbol::PySymbol,
    inner: &symbol::PySymbol,
    path: Vec<usize>,
) -> PyResult<Vec<fit::PyFitMap>> {
    fit::pyfit_at_impl(outer, inner, &path)
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

#[pyfunction(fit_and_apply)]
fn fit_and_apply(
    py: Python,
    variable_creator: PyObject,
    prev: &symbol::PySymbol,
    rule: &rule::PyRule,
) -> PyResult<Vec<(symbol::PySymbol, fit::PyFitMap)>> {
    fit::pyfit_and_apply_impl(py, variable_creator, prev, rule)
}

#[pyfunction(fit_and_apply_at)]
fn fit_at_and_apply(
    py: Python,
    variable_creator: PyObject,
    prev: &symbol::PySymbol,
    rule: &rule::PyRule,
    path: Vec<usize>,
) -> PyResult<Option<(symbol::PySymbol, fit::PyFitMap)>> {
    fit::pyfit_at_and_apply_impl(py, variable_creator, prev, rule, &path)
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
    m.add_wrapped(wrap_pyfunction!(fit_and_apply))?;
    m.add_wrapped(wrap_pyfunction!(fit_at))?;
    m.add_wrapped(wrap_pyfunction!(fit_at_and_apply))?;
    m.add_wrapped(wrap_pyfunction!(apply))?;
    Ok(())
}
