use crate::symbol::PySymbol;
use core;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(name=FitMap,subclass)]
#[derive(Debug)]
pub struct PyFitMap {
    pub variable: HashMap<PySymbol, PySymbol>,
    pub path: Vec<usize>,
}

impl PyFitMap {
    pub fn new<'a>(fitmap: &core::FitMap<'a>) -> PyFitMap {
        PyFitMap {
            path: fitmap.path.clone(),
            variable: fitmap
                .variable
                .iter()
                .map(|(k, v)| (PySymbol::new((*k).clone()), PySymbol::new((*v).clone())))
                .collect(),
        }
    }

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
impl PyObjectProtocol for PyFitMap {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __str__(&self) -> PyResult<String> {
        let mapping = self
            .variable
            .iter()
            .map(|(s, t)| format!("{} -> {}", s.inner, t.inner))
            .collect::<Vec<_>>()
            .join(", ");
        let path = self
            .path
            .iter()
            .map(|p| format!("{}", p))
            .collect::<Vec<_>>()
            .join("/");
        Ok(format!("{} /{}", mapping, path))
    }
}

pub fn pyfit_impl(outer: &PySymbol, inner: &PySymbol) -> PyResult<Vec<PyFitMap>> {
    let fitmaps = core::fit::fit(&outer.inner, &inner.inner);

    let fitmaps = fitmaps.iter().map(PyFitMap::new).collect();

    Ok(fitmaps)
}

pub fn pyfit_at_impl(
    outer: &PySymbol,
    inner: &PySymbol,
    path: &[usize],
) -> PyResult<Vec<PyFitMap>> {
    Ok(core::fit::fit_at(&outer.inner, &inner.inner, path)
        .iter()
        .map(PyFitMap::new)
        .collect())
}
