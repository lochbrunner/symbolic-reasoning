pub mod sample;

use crate::common::op_to_string;
use crate::rule::PyRule;
use core::bag;
use core::rule::Rule;
use pyo3::class::basic::CompareOp;
use pyo3::class::{PyObjectProtocol, PySequenceProtocol};
use pyo3::exceptions::{FileNotFoundError, NotImplementedError, TypeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sample::PySample;
use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::sync::Arc;

#[pyclass(name=BagMeta,subclass)]
pub struct PyBagMeta {
    data: Arc<bag::Meta>,
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
                inner: Arc::new(r.clone()),
                name: l.clone(),
            })
            .collect())
    }

    #[getter]
    fn rule_distribution(&self) -> PyResult<Vec<(u32, u32)>> {
        Ok(self.data.rule_distribution.clone())
    }

    #[getter]
    fn value_distribution(&self) -> PyResult<(u32, u32)> {
        Ok(self.data.value_distribution.clone())
    }

    fn clear_distributions(&mut self) -> PyResult<()> {
        let mut meta = Arc::get_mut(&mut self.data)
            .ok_or(PyErr::new::<TypeError, _>("Can not mutate borrowed bag!"))?;
        for rule in meta.rule_distribution.iter_mut() {
            *rule = (0, 0);
        }
        meta.value_distribution = (0, 0);
        Ok(())
    }

    fn clone_with_distribution(&self, samples: Vec<PySample>) -> PyResult<Self> {
        let mut positive_value = 0;
        let mut negative_value = 0;

        let mut rule_distribution = vec![(0, 0); self.data.rule_distribution.len()];

        for sample in samples.iter() {
            if sample.data.useful {
                positive_value += sample.data.fits.len() as u32;
            } else {
                negative_value += sample.data.fits.len() as u32;
            }
            for fitinfo in sample.data.fits.iter() {
                let (ref mut positive, ref mut negative) =
                    rule_distribution[fitinfo.data.rule_id as usize];
                if fitinfo.data.policy == bag::Policy::Positive {
                    *positive += 1;
                } else {
                    *negative += 1;
                }
            }
        }

        Ok(Self {
            data: Arc::new(bag::Meta {
                rule_distribution,
                value_distribution: (positive_value, negative_value),
                idents: self.data.idents.iter().cloned().collect(),
                rules: self.data.rules.iter().cloned().collect(),
            }),
        })
    }

    fn update_distributions(&mut self, samples: Vec<PySample>) -> PyResult<()> {
        let mut meta = Arc::get_mut(&mut self.data)
            .ok_or(PyErr::new::<TypeError, _>("Can not mutate borrowed bag!"))?;
        // Clear
        for rule in meta.rule_distribution.iter_mut() {
            *rule = (0, 0);
        }
        let mut positive_value = 0;
        let mut negative_value = 0;

        for sample in samples.iter() {
            if sample.data.useful {
                positive_value += sample.data.fits.len() as u32;
            } else {
                negative_value += sample.data.fits.len() as u32;
            }
            for fitinfo in sample.data.fits.iter() {
                let (ref mut positive, ref mut negative) =
                    meta.rule_distribution[fitinfo.data.rule_id as usize];
                if fitinfo.data.policy == bag::Policy::Positive {
                    *positive += 1;
                } else {
                    *negative += 1;
                }
            }
        }

        meta.value_distribution = (positive_value, negative_value);
        Ok(())
    }
}

#[pyclass(name=FitInfo,subclass)]
#[derive(Clone)]
pub struct PyFitInfo {
    /// Starting with 1 for better embedding
    pub data: Arc<bag::FitInfo>,
}

impl PyFitInfo {
    pub fn new(orig: bag::FitInfo) -> PyFitInfo {
        PyFitInfo {
            data: Arc::new(orig),
        }
    }
}

#[pymethods]
impl PyFitInfo {
    #[new]
    fn py_new(rule_id: u32, path: Vec<usize>, positive: bool) -> Self {
        PyFitInfo::new(bag::FitInfo {
            rule_id,
            path,
            policy: bag::Policy::new(positive),
        })
    }

    #[getter]
    fn rule(&self) -> PyResult<u32> {
        Ok(self.data.rule_id)
    }

    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.data.path.clone())
    }

    #[getter]
    fn policy(&self) -> PyResult<f32> {
        Ok(self.data.policy.value())
    }
}

#[pyproto]
impl PyObjectProtocol for PyFitInfo {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", *self.data))
    }

    fn __richcmp__(&self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(*self.data == *other.data),
            CompareOp::Ne => Ok(*self.data != *other.data),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut state = DefaultHasher::new();
        (*self.data).hash(&mut state);
        Ok(state.finish() as isize)
    }
}

#[pyclass(name=Container,subclass)]
#[derive(Clone)]
pub struct PyContainer {
    pub max_depth: u32,
    pub max_spread: u32,
    pub max_size: u32,
    pub samples: Vec<PySample>,
}

#[pymethods]
impl PyContainer {
    #[new]
    fn py_new() -> Self {
        PyContainer {
            max_depth: 0,
            max_spread: 0,
            max_size: 0,
            samples: vec![],
        }
    }

    #[getter]
    fn max_depth(&self) -> PyResult<u32> {
        Ok(self.max_depth)
    }

    #[getter]
    fn max_spread(&self) -> PyResult<u32> {
        Ok(self.max_spread)
    }

    #[getter]
    fn max_size(&self) -> PyResult<u32> {
        Ok(self.max_size)
    }

    #[getter]
    fn samples(&self) -> PyResult<Vec<PySample>> {
        Ok(self.samples.clone())
    }

    fn add_sample(&mut self, sample: PySample) -> PyResult<()> {
        let symbol = &sample.get_initial();
        let size = symbol.size();
        let spread = symbol.max_spread();
        let depth = symbol.depth;
        self.max_depth = cmp::max(depth, self.max_depth);
        self.max_spread = cmp::max(spread, self.max_spread);
        self.max_size = cmp::max(size, self.max_size);
        self.samples.push(sample);
        Ok(())
    }
}

#[pyproto]
impl PySequenceProtocol for PyContainer {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.samples.len())
    }
}

#[pyclass(name=Bag,subclass)]
pub struct PyBag {
    /// Arc<bag::Meta> would be more performing but hard to update
    meta_data: bag::Meta,
    containers: Vec<PyContainer>,
}

#[pymethods]
impl PyBag {
    #[new]
    fn py_new(rules: Vec<&PyTuple>) -> PyResult<Self> {
        let rules = rules
            .into_iter()
            .map(|item| {
                let name = item.get_item(0).extract::<String>()?;
                let rule = (*item.get_item(1).extract::<PyRule>()?.inner).clone();
                Ok((name, rule))
            })
            .collect::<Result<Vec<(std::string::String, Rule)>, PyErr>>()?;
        Ok(PyBag {
            meta_data: bag::Meta {
                idents: vec![],
                rule_distribution: vec![(0, 0); rules.len()],
                value_distribution: (0, 0),
                rules,
            },
            containers: vec![],
        })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<PyBag> {
        let file = File::open(path.clone())
            .map_err(|msg| PyErr::new::<FileNotFoundError, _>(format!("{}: \"{}\"", msg, path)))?;
        let reader = BufReader::new(file);
        let bag = bag::Bag::read_bincode(reader).map_err(PyErr::new::<TypeError, _>)?;

        let meta_data = bag.meta;
        let containers = bag
            .containers
            .into_iter()
            .map(|c| PyContainer {
                max_depth: c.max_depth,
                max_spread: c.max_spread,
                max_size: c.max_size,
                samples: c.samples.into_iter().map(PySample::from).collect(),
            })
            .collect();

        Ok(PyBag {
            meta_data,
            containers,
        })
    }

    fn dump(&self, path: String) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|msg| PyErr::new::<FileNotFoundError, _>(msg.to_string()))?;
        let writer = BufWriter::new(file);
        let meta = self.meta_data.clone();

        let containers: Vec<bag::SampleContainer> = self
            .containers
            .iter()
            .map(|container| bag::SampleContainer {
                max_depth: container.max_depth,
                max_spread: container.max_spread,
                max_size: container.max_size,
                samples: container.samples.iter().map(bag::Sample::from).collect(),
            })
            .collect();

        let bag = bag::Bag { meta, containers };

        bag.write_bincode(writer)
            .map_err(PyErr::new::<TypeError, _>)?;
        Ok(())
    }

    #[getter]
    fn meta(&self) -> PyResult<PyBagMeta> {
        Ok(PyBagMeta {
            data: Arc::new(self.meta_data.clone()),
        })
    }

    #[getter]
    fn containers(&self) -> PyResult<Vec<PyContainer>> {
        Ok(self.containers.clone())
    }

    fn add_container(&mut self, container: PyContainer) -> PyResult<()> {
        self.containers.push(container);
        Ok(())
    }

    fn update_meta(&mut self) -> PyResult<()> {
        let mut idents = bag::extract_idents_from_rules(&self.meta_data.rules, |(_, r)| r);
        let mut rule_distribution: Vec<(u32, u32)> = vec![(0, 0); self.meta_data.rules.len()];
        let mut positive_contributions = 0;
        let mut negative_contributions = 0;
        for container in self.containers.iter() {
            for sample in container.samples.iter() {
                for part in sample.get_initial().iter_bfs() {
                    if !idents.contains(&part.ident) {
                        idents.insert(part.ident.clone());
                    }
                }
                let fits = sample.get_fits();
                if sample.data.useful {
                    positive_contributions += fits.len() as u32;
                } else {
                    negative_contributions += fits.len() as u32;
                }
                for fit in fits.iter() {
                    let (ref mut positive, ref mut negative) =
                        rule_distribution[fit.data.rule_id as usize];
                    if fit.data.policy == bag::Policy::Positive {
                        *positive += 1;
                    } else {
                        *negative += 1;
                    }
                }
            }
        }

        self.meta_data.idents = idents.into_iter().collect();
        self.meta_data.rule_distribution = rule_distribution;
        self.meta_data.value_distribution = (positive_contributions, negative_contributions);

        Ok(())
    }

    fn clear_containers(&mut self) -> PyResult<()> {
        self.containers.clear();
        Ok(())
    }
}
