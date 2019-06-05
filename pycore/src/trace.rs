use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::{DenseApplyInfo, DenseTrace, DenseTraceStep};
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::rc::Rc;

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
        let inner = self.inner.rule.clone();
        Ok(PyRule { inner })
    }

    #[getter]
    fn get_path(&self) -> PyResult<Vec<usize>> {
        let inner = self.inner.path.clone();
        Ok(inner)
    }

    #[getter]
    fn get_initial(&self) -> PyResult<PySymbol> {
        let inner = self.inner.initial.clone();
        Ok(PySymbol { inner })
    }

    #[getter]
    fn get_deduced(&self) -> PyResult<PySymbol> {
        let inner = self.inner.deduced.clone();
        Ok(PySymbol { inner })
    }
}

impl PyApplyInfo {
    // inner: DenseApplyInfo
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
    // cursor: Vec<usize>,
    // trace: Rc<DenseTrace>,
    // pub inner: Calculation<'a>,
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
    trace: Rc<DenseTrace>,
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
            while let Err(_) = s.try_go_sideward() {
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

#[pyclass(name=Trace,subclass)]
pub struct PyTrace {
    inner: Rc<DenseTrace>,
}

#[pymethods]
impl PyTrace {
    #[staticmethod]
    fn load(path: String) -> PyResult<PyTrace> {
        let file = match File::open(path) {
            Err(msg) => Err(PyErr::new::<exceptions::TypeError, _>(msg.to_string())),
            Ok(file) => Ok(file),
        }?;
        let reader = BufReader::new(file);
        let inner = match DenseTrace::read_bincode(reader) {
            Err(msg) => Err(PyErr::new::<exceptions::TypeError, _>(msg.to_string())),
            Ok(trace) => Ok(trace),
        }?;
        let inner = Rc::new(inner);
        Ok(PyTrace { inner })
    }

    fn unroll(&self) -> PyResult<PyTraceIter> {
        Ok(PyTraceIter {
            trace: self.inner.clone(),
            cursor: self.inner.initial_cursor(),
        })
    }
}
