use crate::scenario::PyScenario;
use std::sync::Arc;

use pyo3::exceptions::TypeError;
use pyo3::prelude::*;

use core::bag;

use crate::bag::sample::PySample;
use crate::rule::PyRule;

#[pyclass(name=BagMeta,subclass)]
pub struct PyBagMeta {
    data: Arc<bag::Meta>,
}

impl PyBagMeta {
    pub fn new(meta: &bag::Meta) -> Self {
        PyBagMeta {
            data: Arc::new(meta.clone()),
        }
    }
}

#[pymethods]
impl PyBagMeta {
    #[staticmethod]
    fn from_scenario(scenario: PyScenario) -> PyResult<Self> {
        let meta = bag::Meta::from_scenario(&scenario.inner);
        Ok(Self {
            data: Arc::new(meta),
        })
    }

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
