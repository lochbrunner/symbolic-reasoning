use crate::context::PyContext;
use crate::symbol::PySymbol;
use core::dumper::{dump_latex, dump_plain};
use core::Rule;
use pyo3::class::basic::PyObjectProtocol;
use pyo3::exceptions;
use pyo3::exceptions::{LookupError, TypeError};
use pyo3::prelude::*;
use std::convert::From;
use std::sync::Arc;

/// Python Wrapper for core::Rule
#[pyclass(name=Rule,module="pycore",subclass)]
#[derive(Clone)]
pub struct PyRule {
    pub inner: Arc<Rule>,
}

impl From<&Rule> for PyRule {
    fn from(rule: &Rule) -> Self {
        Self {
            inner: Arc::new(rule.clone()),
        }
    }
}

#[pymethods]
impl PyRule {
    #[new]
    fn py_new(condition: PySymbol, conclusion: PySymbol, name: &str) -> Self {
        Self {
            inner: Arc::new(Rule {
                condition: (*condition.inner).clone(),
                conclusion: (*conclusion.inner).clone(),
                name: name.to_owned(),
            }),
        }
    }

    /// Just uses the first rule.
    #[staticmethod]
    fn parse(context: &PyContext, code: String, name: Option<String>) -> PyResult<PyRule> {
        match Rule::parse(&context.inner, &code) {
            Ok(mut rule) => {
                if let Some(mut rule) = rule.pop() {
                    rule.name = name.unwrap_or(format!("Parsed from {}", code));
                    Ok(PyRule {
                        inner: Arc::new(rule),
                    })
                } else {
                    Err(PyErr::new::<exceptions::ValueError, _>(format!(
                        "Can not parse rule from {}",
                        code
                    )))
                }
            }
            Err(msg) => Err(PyErr::new::<exceptions::TypeError, _>(msg)),
        }
    }

    /// Replace each occurrence of `pattern` with `target` and pads the node with `pad_symbol`
    #[text_signature = "($self, pattern, target, pad_size, pad_symbol /)"]
    fn replace_and_pad(
        &self,
        pattern: String,
        target: String,
        pad_size: u32,
        pad_symbol: PySymbol,
    ) -> PyResult<Self> {
        let pad_symbol = (*pad_symbol.inner).clone();
        let replacer = |orig: &core::Symbol| {
            if orig.ident == pattern {
                let mut new = orig.clone();
                new.ident = target.clone();
                while new.childs.len() < pad_size as usize {
                    new.childs.push(pad_symbol.clone())
                }
                Ok(Some(new))
            } else {
                Ok(None)
            }
        };
        let condition = self.inner.condition.replace::<_, PyErr>(&replacer)?;
        let conclusion = self.inner.conclusion.replace::<_, PyErr>(&replacer)?;
        Ok(PyRule {
            inner: Arc::new(Rule {
                condition,
                conclusion,
                name: self.inner.name.clone(),
            }),
        })
    }

    #[getter]
    fn get_condition(&self) -> PyResult<PySymbol> {
        let inner = self.inner.condition.clone();
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn get_conclusion(&self) -> PyResult<PySymbol> {
        let inner = self.inner.conclusion.clone();
        Ok(PySymbol::new(inner))
    }

    #[getter]
    fn reverse(&self) -> PyResult<PyRule> {
        Ok(PyRule {
            inner: Arc::new(Rule {
                conclusion: self.inner.condition.clone(),
                condition: self.inner.conclusion.clone(),
                name: format!("Reverse of {}", &self.inner.name),
            }),
        })
    }

    /// Dumps the verbose order of operators with equal precedence
    #[getter]
    fn verbose(&self) -> PyResult<String> {
        Ok(format!(
            "{} => {} ",
            dump_plain(&self.inner.condition, &[], true),
            dump_plain(&self.inner.conclusion, &[], true)
        ))
    }

    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(format!(
            "{} \\Rightarrow {} ",
            dump_latex(&self.inner.condition, &[], false),
            dump_latex(&self.inner.conclusion, &[], false)
        ))
    }

    #[getter]
    fn latex_verbose(&self) -> PyResult<String> {
        Ok(format!(
            "{} \\Rightarrow {} ",
            dump_latex(&self.inner.condition, &[], true),
            dump_latex(&self.inner.conclusion, &[], true)
        ))
    }

    #[getter]
    fn get_name(&self) -> PyResult<String> {
        Ok(self.inner.name.clone())
    }

    #[setter]
    fn set_name(&mut self, name: &str) -> PyResult<()> {
        if let Some(rule) = Arc::get_mut(&mut self.inner) {
            rule.name = name.to_owned();
            Ok(())
        } else {
            Err(PyErr::new::<exceptions::ReferenceError, _>(
                "Could not mutable borrow reference of rule".to_owned(),
            ))
        }
    }

    /// Used for pickle
    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        let refs = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            PyErr::new::<TypeError, _>(format!("Can not get mut reference of rule ",))
        })?;
        *refs = bincode::deserialize(&state[..]).map_err(|msg| {
            PyErr::new::<LookupError, _>(format!("Could not deserialize rule\"{:?}\"", msg))
        })?;
        Ok(())
    }
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        bincode::serialize(&*self.inner).map_err(|msg| {
            PyErr::new::<LookupError, _>(format!(
                "Could not serialize rule {}: \"{:?}\"",
                self.inner, msg
            ))
        })
    }
    pub fn __getnewargs__(&self) -> PyResult<(PySymbol, PySymbol, &str)> {
        Ok((
            PySymbol::new(self.inner.condition.clone()),
            PySymbol::new(self.inner.conclusion.clone()),
            &self.inner.name,
        ))
    }
}

#[pyproto]
impl PyObjectProtocol for PyRule {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}
