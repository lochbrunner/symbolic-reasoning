use crate::bag::{PyContainer, PyFitInfo};
use crate::symbol::PySymbol;
use crate::symbol::{make_2darray, make_optional_2darray, PyEmbedding};
use core::dumper::dump_symbol_plain;
use core::io::bag;
use core::symbol::Embedding;
use core::symbol::Symbol;
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::class::basic::PyObjectProtocol;
use pyo3::class::PyMappingProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::PyNumberProtocol;
use std::cmp;
use std::collections::HashMap;
use std::convert::From;
use std::fmt::{Debug, Formatter, Result};
use std::sync::Arc;

/// Must be public as no counter part in core is available
#[derive(Clone, Debug)]
pub struct SampleData {
    initial: PySymbol,
    pub useful: bool,
    pub fits: Vec<PyFitInfo>,
}

#[pyclass(name=Sample,subclass)]
#[derive(Clone)]
pub struct PySample {
    pub data: Arc<SampleData>,
}

impl Debug for PySample {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.data.fmt(f)
    }
}

#[pyproto]
impl PyObjectProtocol for PySample {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self.data))
    }
}

#[pymethods]
impl PySample {
    #[new]
    #[args(initial, fits, useful = "true")]
    fn py_new(initial: PySymbol, fits: Vec<PyFitInfo>, useful: bool) -> Self {
        PySample {
            data: Arc::new(SampleData {
                initial,
                fits,
                useful,
            }),
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

    #[getter]
    fn useful(&self) -> PyResult<bool> {
        Ok(self.data.useful)
    }

    fn create_embedding(
        &self,
        dict: HashMap<String, i16>,
        padding: i16,
        spread: usize,
        max_depth: u32,
        index_map: bool,
        positional_encoding: bool,
    ) -> PyResult<PyEmbedding> {
        let fits = self
            .data
            .fits
            .iter()
            .map(|fit| (*fit.data).clone())
            .collect::<Vec<_>>();
        let embedding = self
            .data
            .initial
            .inner
            .embed(
                &dict,
                padding,
                spread,
                max_depth,
                &fits,
                self.data.useful,
                index_map,
                positional_encoding,
            )
            .map_err(|msg| {
                PyErr::new::<exceptions::KeyError, _>(format!(
                    "Could not embed {}: \"{}\"",
                    self.data.initial.inner, msg
                ))
            })?;
        Ok(PyEmbedding::new(embedding))
    }

    #[args(
        self,
        ident2index,
        padding,
        spread,
        max_depth,
        index_map,
        positional_encoding
    )]
    fn embed(
        &self,
        py: Python,
        dict: HashMap<String, i16>,
        padding: i16,
        spread: usize,
        max_depth: u32,
        index_map: bool,
        positional_encoding: bool,
    ) -> PyResult<(
        Py<PyArray2<i64>>,
        Option<Py<PyArray2<i16>>>,
        Option<Py<PyArray2<i64>>>,
        Py<PyArray1<i64>>,
        Py<PyArray1<f32>>,
        Py<PyArray1<i64>>,
    )> {
        let fits = self
            .data
            .fits
            .iter()
            .map(|fit| (*fit.data).clone())
            .collect::<Vec<_>>();
        let Embedding {
            embedded,
            index_map,
            positional_encoding,
            label,
            policy,
            value,
        } = self
            .data
            .initial
            .inner
            .embed(
                &dict,
                padding,
                spread,
                max_depth,
                &fits,
                self.data.useful,
                index_map,
                positional_encoding,
            )
            .map_err(|msg| {
                PyErr::new::<exceptions::KeyError, _>(format!(
                    "Could not embed {}: \"{}\"",
                    self.data.initial.inner, msg
                ))
            })?;

        let index_map = make_optional_2darray(py, index_map)?;
        let positional_encoding = make_optional_2darray(py, positional_encoding)?;
        let label = label.into_pyarray(py).to_owned();
        let policy = policy.into_pyarray(py).to_owned();
        let value = [value].to_pyarray(py).to_owned(); // value.into_pyarray(py).to_owned();
        let embedded = make_2darray(py, embedded)?;
        Ok((
            embedded,
            index_map,
            positional_encoding,
            label,
            policy,
            value,
        ))
    }
}
impl From<core::io::bag::Sample> for PySample {
    fn from(orig: core::io::bag::Sample) -> Self {
        let core::io::bag::Sample {
            useful,
            initial,
            fits,
        } = orig;
        Self {
            data: Arc::new(SampleData {
                initial: PySymbol::new(initial),
                useful,
                fits: fits.into_iter().map(PyFitInfo::new).collect(),
            }),
        }
    }
}
impl From<&PySample> for core::io::bag::Sample {
    fn from(orig: &PySample) -> Self {
        Self {
            initial: (*orig.data.initial.inner).clone(),
            useful: orig.data.useful,
            fits: orig
                .data
                .fits
                .iter()
                .map(|fit| (*fit.data).clone())
                .collect(),
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

impl PySampleSet {
    fn try_add(&mut self, sample: PySample) -> bool {
        let key = dump_symbol_plain(&(*sample.data.initial.inner), true);

        match self.samples.get_mut(&key) {
            None => {
                self.samples.insert(key, (*sample.data).clone());
            }
            Some(ref mut prev_sample) => {
                // Useful samples dominate
                prev_sample.useful |= sample.data.useful;

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
        true
    }
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
        Ok(self.try_add(sample))
    }

    fn merge(&mut self, other: &PySampleSet) -> PyResult<()> {
        for (key, sample) in other.samples.iter() {
            match self.samples.get_mut(key) {
                None => {
                    self.samples.insert(key.clone(), sample.clone());
                }
                Some(ref mut prev_sample) => {
                    // Useful samples dominate
                    prev_sample.useful |= sample.useful;
                    // Does the fit already exists?
                    // If contradicting use the positive
                    for new_fitinfo in sample.fits.iter().map(|f| &(*f.data)) {
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
        }
        Ok(())
    }

    fn to_container(&self) -> PyResult<PyContainer> {
        let max_size = self
            .samples
            .iter()
            .map(|(_, s)| s.initial.inner.size())
            .fold(0, |p, c| cmp::max(p, c));
        let max_depth = self
            .samples
            .iter()
            .map(|(_, s)| s.initial.inner.depth)
            .fold(0, |p, c| cmp::max(p, c));
        let samples = self
            .samples
            .iter()
            .map(|(_, s)| PySample {
                data: Arc::new((*s).clone()),
            })
            .collect();

        let max_spread = 2;
        Ok(PyContainer {
            max_depth,
            max_spread,
            max_size,
            samples,
        })
    }

    #[staticmethod]
    fn from_container(container: PyContainer) -> PyResult<Self> {
        let mut sample_set = Self {
            samples: HashMap::new(),
        };
        for sample in container.samples.iter() {
            sample_set.add(sample.clone())?;
        }
        Ok(sample_set)
    }

    fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self.samples.keys().cloned().collect())
    }

    fn values(&self) -> PyResult<Vec<PySample>> {
        Ok(self
            .samples
            .values()
            .cloned()
            .map(|v| PySample { data: Arc::new(v) })
            .collect())
    }

    fn items(&self) -> PyResult<Vec<(String, PySample)>> {
        Ok(self
            .samples
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    PySample {
                        data: Arc::new(v.clone()),
                    },
                )
            })
            .collect())
    }
}

#[pyproto]
impl PyMappingProtocol for PySampleSet {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.samples.len())
    }

    fn __getitem__(&self, idx: String) -> PyResult<PySample> {
        if let Some(sample) = self.samples.get(&idx) {
            Ok(PySample {
                data: Arc::new(sample.clone()),
            })
        } else {
            Err(PyErr::new::<exceptions::IndexError, _>(format!(
                "Required index {} is not in range [0, {})",
                idx,
                self.samples.len()
            )))
        }
    }
}

#[pyproto]
impl PyNumberProtocol for PySampleSet {
    fn __iadd__(&mut self, problem: PySample) {
        self.try_add(problem);
    }
}
