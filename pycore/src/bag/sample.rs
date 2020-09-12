use crate::bag::{PyContainer, PyFitInfo};
use crate::symbol::PySymbol;
use core::bag;
use core::dumper::dump_symbol_plain;
use core::symbol::Symbol;
use pyo3::class::PySequenceProtocol;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::convert::From;
use std::sync::Arc;

#[derive(Clone)]
struct SampleData {
    initial: PySymbol,
    fits: Vec<PyFitInfo>,
}

#[pyclass(name=Sample,subclass)]
#[derive(Clone)]
pub struct PySample {
    data: Arc<SampleData>,
}

#[pymethods]
impl PySample {
    #[new]
    fn py_new(initial: PySymbol, fits: Vec<PyFitInfo>) -> Self {
        PySample {
            data: Arc::new(SampleData { initial, fits }),
        }
    }

    #[getter]
    fn initial(&self) -> PyResult<PySymbol> {
        Ok(self.data.initial.clone())
    }

    #[getter]
    fn fits(&self) -> PyResult<Vec<PyFitInfo>> {
        Ok(self.data.fits.clone())
    }
}
impl From<core::bag::Sample> for PySample {
    fn from(orig: core::bag::Sample) -> Self {
        Self {
            data: Arc::new(SampleData {
                initial: PySymbol::new(orig.initial),
                fits: orig.fits.into_iter().map(PyFitInfo::new).collect(),
            }),
        }
    }
}

impl PySample {
    pub fn get_initial(&self) -> &Arc<Symbol> {
        &self.data.initial.inner
    }
    pub fn get_fits(&self) -> &Vec<PyFitInfo> {
        &self.data.fits
    }
}

#[pyclass(name=SampleSet,subclass)]
pub struct PySampleSet {
    samples: HashMap<String, SampleData>,
}

/// Manages creating new samples
/// Avoids duplicates and merges the fit resuĺts in case of collisions
#[pymethods]
impl PySampleSet {
    #[new]
    fn py_new() -> Self {
        Self {
            samples: HashMap::new(),
        }
    }

    /// Returns true if the initial of the sample was not present yet.
    fn add(&mut self, sample: PySample) -> PyResult<bool> {
        let key = dump_symbol_plain(&(*sample.data.initial.inner), true);

        match self.samples.get_mut(&key) {
            None => {
                self.samples.insert(key, (*sample.data).clone());
            }
            Some(ref mut prev_sample) => {
                // Does the fit already exists?
                // If contradicting use the positive
                for new_fitinfo in sample.data.fits.iter().map(|f| &(*f.data)) {
                    match new_fitinfo.compare_many(&(prev_sample.fits), |f| &(*f.data)) {
                        bag::FitCompare::Unrelated => prev_sample.fits.push(PyFitInfo {
                            data: Arc::new(new_fitinfo.clone()),
                        }),
                        bag::FitCompare::Matching => {} // Sample already known
                        bag::FitCompare::Contradicting => {} // Let's try with the positive
                    }
                }
            }
        }

        Ok(true)
    }

    fn to_container(&self) -> PyResult<PyContainer> {
        let samples = self
            .samples
            .iter()
            .map(|(_, s)| PySample {
                data: Arc::new((*s).clone()),
            })
            .collect();
        let max_depth = 0;
        let max_spread = 0;
        let max_size = 0;
        Ok(PyContainer {
            max_depth,
            max_spread,
            max_size,
            samples,
        })
    }
}

#[pyproto]
impl PySequenceProtocol for PySampleSet {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.samples.len())
    }
}
