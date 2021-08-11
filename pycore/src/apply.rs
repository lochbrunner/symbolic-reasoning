use core::FitMap;
use crate::fit::PyFitMap;
use crate::symbol::PySymbol;
use pyo3::prelude::*;
use core::apply;
use std::collections::HashMap;

pub fn pyapply_impl(py: Python, mapping: PyFitMap, variable_creator: PyObject, prev: PySymbol, conclusion: PySymbol) -> PyResult<PySymbol> {

    let fitmap = FitMap{
        variable: HashMap::new(),
        path: vec![],
    };
    let new = apply(&fitmap, || variable_creator.call(py, None, None), &prev.inner, &conclusion.inner );
    // let result = apply();
    unimplemented!();
}
