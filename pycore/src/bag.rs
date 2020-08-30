use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::bag;
use core::rule::Rule;
use pyo3::exceptions::{FileNotFoundError, TypeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cmp;
use std::fs::File;
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
        let symbol = &sample.data.initial.inner;
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

#[pyclass(name=Bag,subclass)]
pub struct PyBag {
    /// Arc<bag::Meta> would be more performing but hard to update
    meta_data: bag::Meta,
    samples_data: Vec<PyContainer>,
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
                rules,
            },
            samples_data: vec![],
        })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<PyBag> {
        let file =
            File::open(path).map_err(|msg| PyErr::new::<FileNotFoundError, _>(msg.to_string()))?;
        let reader = BufReader::new(file);
        let bag = bag::Bag::read_bincode(reader).map_err(PyErr::new::<TypeError, _>)?;

        let meta_data = bag.meta;
        let samples_data = bag
            .samples
            .into_iter()
            .map(|c| PyContainer {
                max_depth: c.max_depth,
                max_spread: c.max_spread,
                max_size: c.max_size,
                samples: c
                    .samples
                    .into_iter()
                    .map(|s| PySample {
                        data: Arc::new(SampleData {
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

    fn dump(&self, path: String) -> PyResult<()> {
        let file = File::create(path)
            .map_err(|msg| PyErr::new::<FileNotFoundError, _>(msg.to_string()))?;
        let writer = BufWriter::new(file);
        let meta = self.meta_data.clone();

        let samples: Vec<bag::SampleContainer> = self
            .samples_data
            .iter()
            .map(|container| bag::SampleContainer {
                max_depth: container.max_depth,
                max_spread: container.max_spread,
                max_size: container.max_size,
                samples: container
                    .samples
                    .iter()
                    .map(|sample| bag::Sample {
                        initial: (*sample.data.initial.inner).clone(),
                        fits: sample
                            .data
                            .fits
                            .iter()
                            .map(|fit| (*fit.data).clone())
                            .collect(),
                    })
                    .collect(),
            })
            .collect();

        let bag = bag::Bag { meta, samples };

        bag.write_bincode(writer)
            .map_err(|msg| PyErr::new::<TypeError, _>(msg.to_string()))?;
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
        Ok(self.samples_data.clone())
    }

    fn add_container(&mut self, container: PyContainer) -> PyResult<()> {
        self.samples_data.push(container);
        Ok(())
    }

    fn update_meta(&mut self) -> PyResult<()> {
        let mut idents = bag::extract_idents_from_rules(&self.meta_data.rules, |(_, r)| r);
        let mut rule_distribution: Vec<(u32, u32)> = vec![(0, 0); self.meta_data.rules.len()];
        for container in self.samples_data.iter() {
            for sample in container.samples.iter() {
                for part in sample.data.initial.inner.iter_bfs() {
                    if !idents.contains(&part.ident) {
                        idents.insert(part.ident.clone());
                    }
                }
                for fit in sample.data.fits.iter() {
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

        Ok(())
    }

    fn clear_containers(&mut self) -> PyResult<()> {
        self.samples_data.clear();
        Ok(())
    }
}
