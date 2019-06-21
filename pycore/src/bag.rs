use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::bag;
use core::{Rule, Symbol};
use std::fs::File;
use std::io::BufReader;
use std::rc::Rc;

use pyo3::exceptions::{FileNotFoundError, TypeError};
use pyo3::prelude::*;

#[pyclass(name=RuleStatistics,subclass)]
#[derive(Clone)]
pub struct PyRuleStatistics {
    rule_ptr: Rc<Rule>,
    fits_inner: usize,
    purposeful_inner: usize,
}

#[pymethods]
impl PyRuleStatistics {
    #[getter]
    fn rule(&self) -> PyResult<PyRule> {
        Ok(PyRule {
            inner: self.rule_ptr.clone(),
        })
    }

    #[getter]
    fn fits(&self) -> PyResult<usize> {
        Ok(self.fits_inner)
    }

    #[getter]
    fn purposeful(&self) -> PyResult<usize> {
        Ok(self.purposeful_inner)
    }
}

struct BagMetaData {
    idents: Vec<String>,
    rules: Vec<PyRuleStatistics>,
}

#[pyclass(name=BagMeta,subclass)]
/// This is a view on the data
pub struct PyBagMeta {
    data: Rc<BagMetaData>,
}

#[pymethods]
impl PyBagMeta {
    #[getter]
    fn idents(&self) -> PyResult<Vec<String>> {
        Ok(self.data.idents.clone())
    }

    #[getter]
    fn rules(&self) -> PyResult<Vec<PyRuleStatistics>> {
        Ok(self.data.rules.clone())
    }
}

struct FitInfoData {
    rule: Rc<Rule>,
    path: Vec<usize>,
    pub purposeful: bool,
}

#[pyclass(name=FitInfo,subclass)]
pub struct PyFitInfo {
    data: Rc<FitInfoData>,
}

#[pymethods]
impl PyFitInfo {
    #[getter]
    fn rule(&self) -> PyResult<PyRule> {
        Ok(PyRule {
            inner: self.data.rule.clone(),
        })
    }

    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.data.path.clone())
    }

    #[getter]
    fn purposeful(&self) -> PyResult<bool> {
        Ok(self.data.purposeful)
    }
}

struct SampleData {
    initial: Rc<Symbol>,
    fits: Vec<Rc<FitInfoData>>,
}

#[pyclass(name=Sample,subclass)]
pub struct PySample {
    data: Rc<SampleData>,
}

#[pymethods]
impl PySample {
    #[getter]
    fn initial(&self) -> PyResult<PySymbol> {
        Ok(PySymbol {
            inner: self.data.initial.clone(),
        })
    }

    #[getter]
    fn fits(&self) -> PyResult<Vec<PyFitInfo>> {
        Ok(self
            .data
            .fits
            .iter()
            .map(|f| PyFitInfo { data: f.clone() })
            .collect())
    }

    #[getter]
    fn purposeful_fits(&self) -> PyResult<Vec<PyFitInfo>> {
        Ok(self
            .data
            .fits
            .iter()
            .filter(|f| f.purposeful)
            .map(|f| PyFitInfo { data: f.clone() })
            .collect())
    }
}

#[pyclass(name=Bag,subclass)]
pub struct PyBag {
    meta_data: Rc<BagMetaData>,
    samples_data: Vec<Rc<SampleData>>,
}

#[pymethods]
impl PyBag {
    #[staticmethod]
    fn load(path: String) -> PyResult<PyBag> {
        let file = match File::open(path) {
            Err(msg) => Err(PyErr::new::<FileNotFoundError, _>(msg.to_string())),
            Ok(file) => Ok(file),
        }?;
        let reader = BufReader::new(file);
        let bag = match bag::Bag::read_bincode(reader) {
            Err(msg) => Err(PyErr::new::<TypeError, _>(msg.to_string())),
            Ok(bag) => Ok(bag),
        }?;
        // Convert bag
        let meta = BagMetaData {
            idents: bag.meta.idents,
            rules: bag
                .meta
                .rules
                .into_iter()
                .map(|s| PyRuleStatistics {
                    rule_ptr: Rc::new(s.rule),
                    fits_inner: s.fits,
                    purposeful_inner: s.purposeful,
                })
                .collect(),
        };
        let meta_data = Rc::new(meta);
        let samples_data = bag
            .samples
            .into_iter()
            .map(|s| {
                Rc::new(SampleData {
                    initial: Rc::new(s.initial),
                    fits: s
                        .fits
                        .into_iter()
                        .map(|f| FitInfoData {
                            rule: Rc::new(f.rule),
                            path: f.path,
                            purposeful: f.purposeful,
                        })
                        .map(|f| Rc::new(f))
                        .collect(),
                })
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
    fn samples(&self) -> PyResult<Vec<PySample>> {
        Ok(self
            .samples_data
            .iter()
            .map(|s| PySample { data: s.clone() })
            .collect())
    }
}
