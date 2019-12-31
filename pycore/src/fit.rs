use crate::symbol::PySymbol;
use core;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name=FitMap,subclass)]
#[derive(Debug)]
pub struct PyFitMap {
    pub variable: HashMap<PySymbol, PySymbol>,
    pub path: Vec<usize>,
}

impl PyFitMap {
    pub fn get_raw(&self) -> core::FitMap {
        core::FitMap {
            path: self.path.clone(),
            variable: self
                .variable
                .iter()
                .map(|(k, v)| (&(*k.inner), &(*v.inner)))
                .collect(),
        }
    }
}

#[pymethods]
impl PyFitMap {
    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.path.clone())
    }

    #[getter]
    fn variable(&self) -> PyResult<HashMap<PySymbol, PySymbol>> {
        Ok(self.variable.clone())
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for PyFitMap {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

pub fn pyfit_impl(outer: &PySymbol, inner: &PySymbol) -> PyResult<Vec<PyFitMap>> {
    let fitmaps = core::fit::fit(&outer.inner, &inner.inner);

    let fitmaps = fitmaps
        .iter()
        .map(|o| PyFitMap {
            path: o.path.clone(),
            variable: o
                .variable
                .iter()
                .map(|(k, v)| (PySymbol::new((*k).clone()), PySymbol::new((*v).clone())))
                .collect(),
        })
        .collect();

    Ok(fitmaps)
}
