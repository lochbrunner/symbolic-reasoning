use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::io::bag::trace::{DenseApplyInfo, DenseTrace, DenseTraceStep};
use pyo3::class::basic::PyObjectProtocol;
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::{FileNotFoundError, TypeError};
use pyo3::prelude::*;
use std::sync::Arc;

use std::fs::File;
use std::io::BufReader;

/// Python Wrapper for core::DenseApplyInfo
#[pyclass(name=ApplyInfo,subclass)]
#[derive(Clone)]
pub struct PyApplyInfo {
    pub inner: DenseApplyInfo,
}

#[pymethods]
impl PyApplyInfo {
    #[getter]
    fn get_rule(&self) -> PyResult<PyRule> {
        let inner = Arc::new(self.inner.rule.clone());
        Ok(PyRule {
            inner,
            name: String::new(),
        })
    }

    #[getter]
    fn get_path(&self) -> PyResult<Vec<usize>> {
        let inner = self.inner.path.clone();
        Ok(inner)
    }

    #[getter]
    fn get_initial(&self) -> PyResult<PySymbol> {
        let inner = self.inner.initial.clone();
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn get_deduced(&self) -> PyResult<PySymbol> {
        let inner = self.inner.deduced.clone();
        Ok(PySymbol::new(inner))
    }
}

#[pyproto]
impl PyObjectProtocol for PyApplyInfo {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __str__(&self) -> PyResult<String> {
        let inner = &self.inner;
        Ok(format!(
            "{} -> {} ({} @ {:?})",
            inner.initial, inner.deduced, inner.rule, inner.path
        ))
    }
}

#[pyclass(name=TraceStep,subclass)]
pub struct PyTraceStep {
    pub inner: DenseTraceStep,
}

#[pymethods]
impl PyTraceStep {}

#[pyclass(name=Calculation,subclass)]
#[derive(Clone)]
pub struct PyCalculation {
    pub steps: Vec<PyApplyInfo>,
}

#[pymethods]
impl PyCalculation {
    #[getter]
    fn get_steps(&self) -> PyResult<Vec<PyApplyInfo>> {
        Ok(self.steps.clone())
    }
}

#[pyclass(name=TraceIter)]
pub struct PyTraceIter {
    cursor: Vec<usize>,
    trace: Arc<DenseTrace>,
}

impl PyTraceIter {
    #[inline]
    fn get_steps(&self) -> Vec<PyApplyInfo> {
        // Extract item
        let mut step = &self.trace.stages;
        let mut steps = vec![];
        for i in self.cursor.iter() {
            steps.push(&step[*i].info);
            step = &step[*i].successors;
        }
        steps
            .into_iter()
            .cloned()
            .map(|inner| PyApplyInfo { inner })
            .collect()
    }

    #[inline]
    fn try_go_sideward(&mut self) -> Result<(), ()> {
        if self.cursor.is_empty() {
            return Err(());
        }
        // Check we can go sideward
        let mut current_stage = &self.trace.stages;
        // Go to second last
        if self.cursor.len() > 1 {
            for i in self.cursor.iter().take(self.cursor.len() - 1) {
                current_stage = &current_stage[*i].successors;
            }
        }
        if current_stage.len() > 1 + *self.cursor.last().unwrap() {
            *self.cursor.last_mut().unwrap() += 1;
            Ok(())
        } else {
            // Go one up to the next: Recursion
            Err(())
        }
    }
    #[inline]
    fn go_to_ground(&mut self) {
        let mut current_stage = &self.trace.stages;
        for i in self.cursor.iter() {
            current_stage = &current_stage[*i].successors;
        }

        while !current_stage.is_empty() {
            self.cursor.push(0);
            current_stage = &current_stage[0].successors;
        }
    }
}

#[pyproto]
impl PyIterProtocol for PyTraceIter {
    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<PyCalculation>> {
        Ok(if s.cursor.is_empty() {
            None
        } else {
            let steps = s.get_steps();
            while s.try_go_sideward().is_err() {
                // Go deeper
                if s.cursor.is_empty() {
                    break;
                } else {
                    s.cursor.pop();
                }
            }
            // empty cursor indicates end of tree
            if !s.cursor.is_empty() {
                s.go_to_ground();
            }
            Some(PyCalculation { steps })
        })
    }

    fn __iter__(s: PyRefMut<Self>) -> PyResult<PyTraceIter> {
        Ok(PyTraceIter {
            trace: s.trace.clone(),
            cursor: s.trace.initial_cursor(),
        })
    }
}

#[pyclass(name=StepsIter)]
pub struct PyStepsIter {
    cursors: Vec<Vec<usize>>,
    trace: Arc<DenseTrace>,
}

impl PyStepsIter {
    fn get_node<'a>(&'a self, cursor: &[usize]) -> &'a DenseTraceStep {
        let mut current_stage = &self.trace.stages;
        for i in cursor.iter().take(cursor.len() - 1) {
            current_stage = &current_stage[*i].successors;
        }
        &current_stage[*cursor.last().unwrap()]
    }
}

#[pyproto]
impl PyIterProtocol for PyStepsIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PyStepsIter> {
        Ok(PyStepsIter {
            trace: s.trace.clone(),
            cursors: s
                .trace
                .stages
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i])
                .collect(),
        })
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<PyApplyInfo>> {
        Ok(if s.cursors.is_empty() {
            None
        } else {
            let cursor = s.cursors.pop().unwrap();
            let node = s.get_node(&cursor);
            let apply_info = PyApplyInfo {
                inner: node.info.clone(),
            };
            let mut new_cursors = Vec::new();
            for (i, _) in node.successors.iter().enumerate() {
                let mut cursor = cursor.clone();
                cursor.push(i);
                new_cursors.push(cursor);
            }
            s.cursors.extend(new_cursors);
            Some(apply_info)
        })
    }
}

#[pyclass(name=Meta,subclass)]
pub struct PyMeta {
    trace: Arc<DenseTrace>,
}

#[pymethods]
impl PyMeta {
    #[getter]
    fn used_idents(&self) -> PyResult<Vec<String>> {
        Ok(self.trace.meta.used_idents.iter().cloned().collect())
    }
    #[getter]
    fn rules(&self) -> PyResult<Vec<PyRule>> {
        Ok(self
            .trace
            .meta
            .rules
            .iter()
            .map(|(n, r)| PyRule {
                inner: Arc::new(r.clone()),
                name: n.clone(),
            })
            .collect())
    }
}

#[pyclass(name=Trace,subclass,dict)]
pub struct PyTrace {
    inner: Arc<DenseTrace>,
}

#[pymethods]
impl PyTrace {
    #[staticmethod]
    fn load(path: String) -> PyResult<PyTrace> {
        let file =
            File::open(path).map_err(|msg| PyErr::new::<FileNotFoundError, _>(msg.to_string()))?;
        let reader = BufReader::new(file);
        let inner = DenseTrace::read_bincode(reader).map_err(PyErr::new::<TypeError, _>)?;
        let inner = Arc::new(inner);
        Ok(PyTrace { inner })
    }

    #[getter]
    fn unroll(&self) -> PyResult<PyTraceIter> {
        Ok(PyTraceIter {
            trace: self.inner.clone(),
            cursor: self.inner.initial_cursor(),
        })
    }

    #[getter]
    fn all_steps(&self) -> PyResult<PyStepsIter> {
        Ok(PyStepsIter {
            trace: self.inner.clone(),
            cursors: self
                .inner
                .stages
                .iter()
                .enumerate()
                .map(|(i, _)| vec![i])
                .collect(),
        })
    }

    #[getter]
    fn meta(&self) -> PyResult<PyMeta> {
        Ok(PyMeta {
            trace: self.inner.clone(),
        })
    }
}
