use core::io::Scenario;
use pyo3::exceptions;
use pyo3::prelude::*;
use std::rc::Rc;

/// Python Wrapper for core::io::Scenario
#[pyclass(name=Scenario,subclass)]
pub struct PyScenario {
    pub inner: Rc<Scenario>,
}

#[pymethods]
impl PyScenario {
    #[staticmethod]
    fn load(code: String) -> PyResult<PyScenario> {
        match Scenario::load_from_yaml(&code) {
            Ok(scenario) => Ok(PyScenario {
                inner: Rc::new(scenario),
            }),
            Err(msg) => Err(PyErr::new::<exceptions::IOError, _>(msg)),
        }
    }
}
