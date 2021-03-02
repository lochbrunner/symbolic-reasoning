pub mod fitinfo;
pub mod meta;
pub mod sample;

use crate::bag::sample::PySample;
use crate::rule::PyRule;
use crate::scenario::PyScenario;
use core::io::bag;
use core::rule::Rule;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::class::PySequenceProtocol;
use pyo3::exceptions::{FileNotFoundError, TypeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::cmp;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub use crate::bag::meta::PyBagMeta;
pub use fitinfo::PyFitInfo;

#[pyclass(name=Container,subclass)]
#[derive(Clone, Debug)]
pub struct PyContainer {
    pub max_depth: u32,
    pub max_spread: u32,
    pub max_size: u32,
    pub samples: Vec<PySample>,
}

#[pyproto]
impl PyObjectProtocol for PyContainer {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[pymethods]
impl PyContainer {
    #[new]
    fn py_new() -> Self {
        Self {
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
                let rule = (*item.get_item(1).extract::<PyRule>()?.inner).clone();
                Ok(rule)
            })
            .collect::<Result<Vec<Rule>, PyErr>>()?;
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
    #[args(scenario, ignore_declaration = true)]
    fn from_scenario(scenario: &PyScenario, ignore_declaration: bool) -> PyResult<Self> {
        Ok(Self {
            meta_data: bag::Meta::from_scenario(&scenario.inner, ignore_declaration),
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
            .map_err(PyErr::new::<TypeError, _>)
    }

    #[getter]
    fn meta(&self) -> PyResult<PyBagMeta> {
        Ok(PyBagMeta::new(&self.meta_data))
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
        let mut idents = bag::extract_idents_from_rules(&self.meta_data.rules, |r| r);
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
