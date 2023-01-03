use crate::rule::PyRule;
use crate::symbol::PySymbol;
use core::Symbol;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions::LookupError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;

#[pyclass(name=FitMap,module="pycore",subclass)]
#[derive(Debug)]
pub struct PyFitMap {
    pub variable: HashMap<PySymbol, PySymbol>,
    pub path: Vec<usize>,
}

impl PyFitMap {
    pub fn new<'a>(fitmap: &core::FitMap<'a>) -> PyFitMap {
        PyFitMap {
            path: fitmap.path.clone(),
            variable: fitmap
                .variable
                .iter()
                .map(|(k, v)| (PySymbol::new((*k).clone()), PySymbol::new((*v).clone())))
                .collect(),
        }
    }

    pub fn get_raw(&self) -> core::FitMap {
        core::FitMap {
            path: self.path.clone(),
            variable: self
                .variable
                .iter()
                .map(|(k, v)| (&(*k.inner), &(*v.inner)))
                .collect(),
        }
    }
}

#[derive(Deserialize, Serialize, Default, Debug)]
struct PyFitMapData {
    pub variable: HashMap<Symbol, Symbol>,
    pub path: Vec<usize>,
}

#[pymethods]
impl PyFitMap {
    #[new]
    fn py_new() -> Self {
        Self {
            variable: HashMap::new(),
            path: Vec::new(),
        }
    }

    #[getter]
    fn path(&self) -> PyResult<Vec<usize>> {
        Ok(self.path.clone())
    }

    #[getter]
    fn variable(&self) -> PyResult<HashMap<PySymbol, PySymbol>> {
        Ok(self.variable.clone())
    }

    // Used for pickle
    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let data: PyFitMapData = bincode::deserialize(&state[..]).map_err(|msg| {
            PyErr::new::<LookupError, _>(format!("Could not deserialize PyFitMap\"{:?}\"", msg))
        })?;
        let PyFitMapData { path, variable } = data;
        self.path = path;
        self.variable = variable
            .into_iter()
            .map(|(a, b)| (PySymbol::new(a), PySymbol::new(b)))
            .collect();
        Ok(())
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        let data = PyFitMapData {
            variable: self
                .variable
                .iter()
                .map(|(a, b)| ((*a.inner).clone(), (*b.inner).clone()))
                .collect(),
            path: self.path.clone(),
        };
        bincode::serialize(&data).map_err(|msg| {
            PyErr::new::<LookupError, _>(format!(
                "Could not serialize symbol {:?}: \"{:?}\"",
                data, msg
            ))
        })
    }
}

#[pyproto]
impl PyObjectProtocol for PyFitMap {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __str__(&self) -> PyResult<String> {
        let mapping = self
            .variable
            .iter()
            .map(|(s, t)| format!("{} -> {}", s.inner, t.inner))
            .collect::<Vec<_>>()
            .join(", ");
        let path = self
            .path
            .iter()
            .map(|p| format!("{}", p))
            .collect::<Vec<_>>()
            .join("/");
        Ok(format!("{} /{}", mapping, path))
    }
}

/// Tries to fit the inner into the outer symbol.
/// Returns a list of possible mappings.
#[pyfunction]
#[text_signature = "(outer, inner, /)"]
fn fit(outer: &PySymbol, inner: &PySymbol) -> PyResult<Vec<PyFitMap>> {
    Ok(core::fit::fit(&outer.inner, &inner.inner)
        .iter()
        .map(PyFitMap::new)
        .collect())
}

/// Tries to fit the inner into the outer symbol at a specified path.
/// Returns a list of possible mappings.
#[pyfunction]
#[text_signature = "(outer, inner, path /)"]
fn fit_at(outer: &PySymbol, inner: &PySymbol, path: Vec<usize>) -> PyResult<Option<PyFitMap>> {
    Ok(core::fit::fit_at(&outer.inner, &inner.inner, &path)
        .iter()
        .map(PyFitMap::new)
        .next())
}

#[pyfunction]
#[text_signature = "(outer, variable_creator, orig, rule, /)"]
fn fit_and_apply(
    py: Python,
    variable_creator: PyObject,
    orig: &PySymbol,
    rule: &PyRule,
) -> PyResult<Vec<(PySymbol, PyFitMap)>> {
    core::fit::fit(&orig.inner, &rule.inner.condition)
        .into_iter()
        .map(|mapping| {
            let r = core::apply::<_, PyErr>(
                &mapping,
                || {
                    let obj = variable_creator.call(py, PyTuple::empty(py), None)?;
                    let symbol: PySymbol = obj.extract(py)?;
                    Ok((*symbol.inner).clone())
                },
                &orig.inner,
                &rule.inner.conclusion,
            );
            (r, mapping)
        })
        .map(|(r, m)| match r {
            Ok(r) => Ok((PySymbol::new(r), PyFitMap::new(&m))),
            Err(e) => Err(e),
        })
        .collect::<Result<Vec<_>, _>>()
}

#[pyfunction]
#[text_signature = "(variable_creator, orig, rule, path, /)"]
fn fit_at_and_apply(
    py: Python,
    variable_creator: PyObject,
    orig: &PySymbol,
    rule: &PyRule,
    path: Vec<usize>,
) -> PyResult<Option<(PySymbol, PyFitMap)>> {
    match core::fit::fit_at(&orig.inner, &rule.inner.condition, &path)
        .into_iter()
        .map(|mapping| {
            let r = core::apply::<_, PyErr>(
                &mapping,
                || {
                    let obj = variable_creator.call(py, PyTuple::empty(py), None)?;
                    let symbol: PySymbol = obj.extract(py)?;
                    Ok((*symbol.inner).clone())
                },
                &orig.inner,
                &rule.inner.conclusion,
            );
            (r, mapping)
        })
        .map(|(r, m)| match r {
            Ok(r) => Ok((PySymbol::new(r), PyFitMap::new(&m))),
            Err(e) => Err(e),
        })
        .next()
    {
        None => Ok(None),
        Some(v) => match v {
            Ok(v) => Ok(Some(v)),
            Err(e) => Err(e),
        },
    }
}

/// Registers all functions and classes regarding fitting.
pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(fit))?;
    m.add_wrapped(wrap_pyfunction!(fit_and_apply))?;
    m.add_wrapped(wrap_pyfunction!(fit_at))?;
    m.add_wrapped(wrap_pyfunction!(fit_at_and_apply))?;
    m.add_class::<PyFitMap>()?;
    Ok(())
}
