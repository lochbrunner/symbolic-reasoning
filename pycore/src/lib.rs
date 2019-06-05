use pyo3::prelude::*;

mod context;
mod rule;
mod symbol;
mod trace;

#[pymodule]
fn pycore(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<context::PyDeclaration>()?;
    m.add_class::<context::PyContext>()?;
    m.add_class::<symbol::PySymbol>()?;
    m.add_class::<rule::PyRule>()?;
    m.add_class::<trace::PyApplyInfo>()?;
    m.add_class::<trace::PyTraceStep>()?;
    m.add_class::<trace::PyTrace>()?;
    Ok(())
}
