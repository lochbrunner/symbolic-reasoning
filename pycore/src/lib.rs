use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod bag;
mod context;
mod fit;
mod rule;
mod scenario;
mod symbol;
mod symbol_builder;
mod trace;

#[pyfunction]
fn fit(outer: &symbol::PySymbol, inner: &symbol::PySymbol) -> PyResult<Vec<fit::PyFitMap>> {
    fit::pyfit_impl(outer, inner)
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
    m.add_class::<bag::PyRuleStatistics>()?;
    m.add_class::<bag::PySample>()?;
    m.add_class::<scenario::PyScenario>()?;
    m.add_class::<fit::PyFitMap>()?;
    m.add_wrapped(wrap_pyfunction!(fit))?;
    Ok(())
}
