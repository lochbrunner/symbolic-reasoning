use pyo3::prelude::*;

mod apply;
mod bag;
mod common;
mod context;
mod fit;
mod rule;
mod scenario;
mod solver_trace;
mod symbol;
mod symbol_builder;
mod trace;

#[macro_use]
extern crate serde_derive;

#[pymodule]
fn pycore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<context::PyDeclaration>()?;
    m.add_class::<context::PyContext>()?;
    m.add_class::<symbol::PyDecoration>()?;
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
    m.add_class::<bag::sample::PySample>()?;
    m.add_class::<bag::sample::PySampleSet>()?;
    m.add_class::<scenario::PyScenario>()?;
    m.add_class::<scenario::PyScenarioProblems>()?;
    solver_trace::register(m)?;
    fit::register(m)?;
    apply::register(m)?;
    Ok(())
}
