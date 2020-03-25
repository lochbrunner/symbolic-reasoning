use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::bag;
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;

use pyo3::exceptions::{FileNotFoundError, TypeError};
use pyo3::prelude::*;

#[pyclass(name=BagMeta,subclass)]
pub struct PyBagMeta {
    data: Rc<bag::Meta>,
}

#[pymethods]
impl PyBagMeta {
    #[getter]
    fn idents(&self) -> PyResult<Vec<String>> {
        Ok(self.data.idents.clone())
    }

    #[getter]
    fn rules(&self) -> PyResult<Vec<PyRule>> {
        Ok(self
            .data
            .rules
            .iter()
            .map(|(l, r)| PyRule {
                inner: Rc::new(r.clone()),
                name: l.clone(),
            })
            .collect())
    }

    #[getter]
    fn rule_distribution(&self) -> PyResult<Vec<u32>> {
        Ok(self.data.rule_distribution.clone())
    }
}

#[pyclass(name=FitInfo,subclass)]
#[derive(Clone)]
pub struct PyFitInfo {
    /// Starting with 1 for better embedding
    pub data: Rc<bag::FitInfo>,
}

impl PyFitInfo {
    pub fn new(orig: bag::FitInfo) -> PyFitInfo {
        PyFitInfo {
            data: Rc::new(orig),
        }
    }
}

#[pymethods]
impl PyFitInfo {
    #[getter]
    fn rule(&self) -> PyResult<u32> {
        Ok(self.data.rule_id)
    }

    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.data.path.clone())
    }
}

struct SampleData {
    initial: PySymbol,
    fits: Vec<PyFitInfo>,
}

#[pyclass(name=Sample,subclass)]
#[derive(Clone)]
pub struct PySample {
    data: Rc<SampleData>,
}

#[pymethods]
impl PySample {
    #[getter]
    fn initial(&self) -> PyResult<PySymbol> {
        Ok(self.data.initial.clone())
    }

    #[getter]
    fn fits(&self) -> PyResult<Vec<PyFitInfo>> {
        Ok(self.data.fits.clone())
    }
}

#[pyclass(name=Container,subclass)]
#[derive(Clone)]
pub struct PyContainer {
    pub max_depth: u32,
    pub max_spread: u32,
    pub samples: Vec<PySample>,
}

#[pymethods]
impl PyContainer {
    #[getter]
    fn max_depth(&self) -> PyResult<u32> {
        Ok(self.max_depth)
    }

    #[getter]
    fn max_spread(&self) -> PyResult<u32> {
        Ok(self.max_spread)
    }

    #[getter]
    fn samples(&self) -> PyResult<Vec<PySample>> {
        Ok(self.samples.clone())
    }
}

#[pyclass(name=Bag,subclass)]
pub struct PyBag {
    meta_data: Rc<bag::Meta>,
    samples_data: Vec<PyContainer>,
}

#[pymethods]
impl PyBag {
    #[staticmethod]
    fn load(path: String) -> PyResult<PyBag> {
        let file =
            File::open(path).map_err(|msg| PyErr::new::<FileNotFoundError, _>(msg.to_string()))?;
        let reader = BufReader::new(file);
        let bag = bag::Bag::read_bincode(reader).map_err(PyErr::new::<TypeError, _>)?;

        let meta_data = Rc::new(bag.meta);
        let samples_data = bag
            .samples
            .into_iter()
            .map(|c| PyContainer {
                max_depth: c.max_depth,
                max_spread: c.max_spread,
                samples: c
                    .samples
                    .into_iter()
                    .map(|s| PySample {
                        data: Rc::new(SampleData {
                            initial: PySymbol::new(s.initial),
                            fits: s.fits.into_iter().map(PyFitInfo::new).collect(),
                        }),
                    })
                    .collect(),
            })
            .collect();

        Ok(PyBag {
            meta_data,
            samples_data,
        })
    }

    #[getter]
    fn meta(&self) -> PyResult<PyBagMeta> {
        Ok(PyBagMeta {
            data: self.meta_data.clone(),
        })
    }

    #[getter]
    fn samples(&self) -> PyResult<Vec<PyContainer>> {
        Ok(self.samples_data.clone())
    }
}
