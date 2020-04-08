use crate::bag::PyFitInfo;
use crate::context::PyContext;
use core::dumper::Decoration;
use core::dumper::{dump_latex, dump_symbol_plain};
use core::Symbol;
use ndarray::Array;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::class::basic::{CompareOp, PyObjectProtocol};
use pyo3::class::iter::PyIterProtocol;
use pyo3::exceptions::{IndexError, KeyError, NotImplementedError, TypeError};
use pyo3::gc::{PyGCProtocol, PyVisit};
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

type Path = Vec<usize>;

/// Python Wrapper for core::dumper::Decoration
#[pyclass(name=Decoration,subclass)]
#[derive(Clone)]
pub struct PyDecoration {
    pub path: Path,
    pub pre: String,
    pub post: String,
}

#[pymethods]
impl PyDecoration {
    #[new]
    fn py_new(path: Path, pre: String, post: String) -> Self {
        PyDecoration { path, pre, post }
    }
}

/// Python Wrapper for core::Symbol
#[pyclass(name=Symbol,subclass)]
#[derive(PartialEq)]
pub struct PySymbol {
    pub inner: Rc<Symbol>,
    pub attributes: HashMap<String, PyObject>,
}

impl Hash for PySymbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl Clone for PySymbol {
    fn clone(&self) -> Self {
        PySymbol {
            inner: self.inner.clone(),
            attributes: HashMap::new(),
        }
    }
}

impl PySymbol {
    pub fn new(symbol: Symbol) -> PySymbol {
        PySymbol {
            inner: Rc::new(symbol),
            attributes: HashMap::new(),
        }
    }
}

impl Eq for PySymbol {}

impl PySymbol {
    fn try_release_attribute(&mut self, name: &str) {
        if let Some((_, obj)) = self.attributes.remove_entry(name) {
            let gil = GILGuard::acquire();
            let py = gil.python();
            py.release(obj);
        }
    }
}

#[pyclass(name=SymbolBfsIter)]
#[derive(Clone)]
pub struct PySymbolBfsIter {
    pub parent: Rc<Symbol>,
    pub queue: VecDeque<Symbol>,
}

impl PySymbolBfsIter {
    pub fn new(symbol: &Rc<Symbol>) -> PySymbolBfsIter {
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back((*symbol.clone()).clone());
        PySymbolBfsIter {
            parent: symbol.clone(),
            queue,
        }
    }
}

#[pyproto]
impl PyIterProtocol for PySymbolBfsIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolBfsIter> {
        Ok(PySymbolBfsIter::new(&s.parent))
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<PySymbol>> {
        match s.queue.pop_front() {
            None => Ok(None),
            Some(current) => {
                for child in current.childs.iter() {
                    s.queue.push_back(child.clone());
                }
                Ok(Some(PySymbol::new(current)))
            }
        }
    }
}

#[pyclass(name=SymbolDfsIter)]
#[derive(Clone)]
pub struct PySymbolDfsIter {
    pub parent: Rc<Symbol>,
    pub stack: Vec<Symbol>,
}

#[pyproto]
impl PyIterProtocol for PySymbolDfsIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolDfsIter> {
        Ok(PySymbolDfsIter {
            parent: s.parent.clone(),
            stack: vec![(*s.parent).clone()],
        })
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<PySymbol>> {
        match s.stack.pop() {
            None => Ok(None),
            Some(current) => {
                for child in current.childs.iter() {
                    s.stack.push(child.clone());
                }
                Ok(Some(PySymbol::new(current)))
            }
        }
    }
}

#[pyclass(name=SymbolAndPathIter)]
#[derive(Clone)]
pub struct PySymbolAndPathDfsIter {
    pub parent: Rc<Symbol>,
    pub stack: Vec<Path>,
}

#[pyproto]
impl PyIterProtocol for PySymbolAndPathDfsIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolAndPathDfsIter> {
        Ok(PySymbolAndPathDfsIter {
            parent: s.parent.clone(),
            stack: vec![vec![]],
        })
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<(Path, PySymbol)>> {
        match s.stack.pop() {
            None => Ok(None),
            Some(path) => {
                let symbol = s
                    .parent
                    .at(&path)
                    .unwrap_or_else(|| panic!("part at path: {:?}", path))
                    .clone();
                for (i, _) in symbol.childs.iter().enumerate() {
                    s.stack.push([&path[..], &[i]].concat());
                }
                Ok(Some((path, PySymbol::new(symbol))))
            }
        }
    }
}

#[pyclass(name=SymbolAndPathIter)]
#[derive(Clone)]
pub struct PySymbolAndPathBfsIter {
    pub parent: Rc<Symbol>,
    pub queue: VecDeque<Path>,
}

impl PySymbolAndPathBfsIter {
    pub fn new(symbol: &Rc<Symbol>) -> PySymbolAndPathBfsIter {
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back(vec![]);
        PySymbolAndPathBfsIter {
            parent: symbol.clone(),
            queue,
        }
    }
}

#[pyproto]
impl PyIterProtocol for PySymbolAndPathBfsIter {
    fn __iter__(s: PyRefMut<Self>) -> PyResult<PySymbolAndPathBfsIter> {
        Ok(PySymbolAndPathBfsIter::new(&s.parent))
    }

    fn __next__(mut s: PyRefMut<Self>) -> PyResult<Option<(Path, PySymbol)>> {
        match s.queue.pop_front() {
            None => Ok(None),
            Some(path) => {
                let current = s
                    .parent
                    .at(&path)
                    .unwrap_or_else(|| panic!("part at path: {:?}", path))
                    .clone();
                for (i, _) in current.childs.iter().enumerate() {
                    s.queue.push_back([&path[..], &[i]].concat());
                }
                Ok(Some((path, PySymbol::new(current))))
            }
        }
    }
}

#[pymethods]
impl PySymbol {
    #[staticmethod]
    #[text_signature = "(context, code, /)"]
    fn parse(context: &PyContext, code: String) -> PyResult<PySymbol> {
        let inner = Symbol::parse(&context.inner, &code).map_err(PyErr::new::<TypeError, _>)?;
        Ok(PySymbol::new(inner))
    }

    #[staticmethod]
    #[text_signature = "(ident, fixed, /)"]
    #[args(fixed = false)]
    fn variable(ident: &str, fixed: bool) -> PyResult<PySymbol> {
        Ok(PySymbol::new(Symbol::new_variable(ident, fixed)))
    }

    #[getter]
    fn ident(&self) -> PyResult<String> {
        Ok(self.inner.ident.clone())
    }

    #[text_signature = "($self, path, /)"]
    fn at(&self, path: Path) -> PyResult<PySymbol> {
        match self.inner.at(&path) {
            None => Err(PyErr::new::<IndexError, _>("Index is out of bound")),
            Some(item) => Ok(PySymbol::new(item.clone())),
        }
    }

    #[getter]
    fn parts_bfs(&self) -> PyResult<PySymbolBfsIter> {
        Ok(PySymbolBfsIter::new(&self.inner))
    }

    #[getter]
    fn parts_dfs(&self) -> PyResult<PySymbolDfsIter> {
        Ok(PySymbolDfsIter {
            parent: self.inner.clone(),
            stack: vec![(*self.inner).clone()],
        })
    }

    #[getter]
    fn parts_dfs_with_path(&self) -> PyResult<PySymbolAndPathDfsIter> {
        Ok(PySymbolAndPathDfsIter {
            parent: self.inner.clone(),
            stack: vec![vec![]],
        })
    }

    #[getter]
    fn parts_bfs_with_path(&self) -> PyResult<PySymbolAndPathBfsIter> {
        Ok(PySymbolAndPathBfsIter::new(&self.inner))
    }

    /// Dumps the verbose order of operators with equal precedence
    #[getter]
    fn verbose(&self) -> PyResult<String> {
        Ok(dump_symbol_plain(&self.inner, true))
    }

    /// LaTeX representation of that node
    #[getter]
    fn latex(&self) -> PyResult<String> {
        Ok(dump_latex(&self.inner, vec![], false))
    }

    /// LaTeX representation of that node with brackets everywhere
    #[getter]
    fn latex_verbose(&self) -> PyResult<String> {
        Ok(dump_latex(&self.inner, vec![], true))
    }

    /// The node as a tree
    #[getter]
    fn tree(&self) -> PyResult<String> {
        Ok(self.inner.print_tree())
    }

    #[getter]
    fn childs(&self) -> PyResult<Vec<PySymbol>> {
        Ok(self
            .inner
            .childs
            .iter()
            .map(|s| PySymbol::new(s.clone()))
            .collect())
    }

    /// Unrolls the symbol tree using breath first traversing
    /// u16 is not supported by pytorch
    fn embed(
        &self,
        py: Python,
        dict: HashMap<String, i16>,
        padding: i16,
        spread: usize,
        fits: Vec<PyFitInfo>,
    ) -> PyResult<(Py<PyArray1<i64>>, Py<PyArray2<i16>>, Py<PyArray1<i64>>)> {
        let fits = fits
            .into_iter()
            .map(|fit| (*fit.data).clone())
            .collect::<Vec<_>>();
        let (embedding, indices, label) =
            self.inner
                .embed(&dict, padding, spread, &fits)
                .map_err(|msg| {
                    PyErr::new::<KeyError, _>(format!(
                        "Could not embed {}: \"{}\"",
                        self.inner, msg
                    ))
                })?;

        let indices = Array::from_shape_vec(
            (indices.len(), indices[0].len()),
            indices
                .into_iter()
                .flat_map(|row| row.into_iter())
                .collect(),
        )
        .map_err(|msg| PyErr::new::<TypeError, _>(msg.to_string()))?;
        let indices = indices.into_pyarray(py).to_owned();
        let label = label.into_pyarray(py).to_owned();
        let embedding = embedding.into_pyarray(py).to_owned();
        Ok((embedding, indices, label))
    }

    #[getter]
    fn depth(&self) -> PyResult<u32> {
        Ok(self.inner.depth)
    }

    #[getter]
    fn size(&self) -> PyResult<u32> {
        Ok(self.inner.size())
    }

    /// Assumes spread of 2.
    #[getter]
    fn density(&self) -> PyResult<f32> {
        Ok(self.inner.density())
    }

    #[getter]
    fn fixed(&self) -> PyResult<bool> {
        Ok(self.inner.fixed())
    }

    #[getter]
    fn only_root(&self) -> PyResult<bool> {
        Ok(self.inner.only_root())
    }

    #[text_signature = "($self, /)"]
    fn clone(&self) -> PyResult<PySymbol> {
        Ok(PySymbol {
            inner: self.inner.clone(),
            attributes: HashMap::new(),
        })
    }

    #[text_signature = "($self, padding, spread, depth, /)"]
    fn pad(&mut self, padding: String, spread: u32, depth: u32) -> PyResult<()> {
        match Rc::get_mut(&mut self.inner) {
            None => Err(PyErr::new::<TypeError, _>(format!(
                "Can not get mut reference of symbol {}",
                self.inner
            ))),
            Some(parent) => {
                for level in 0..depth {
                    for node in parent.iter_level_mut(level) {
                        let nc = node.childs.len() as u32;
                        for _ in nc..spread {
                            node.childs.push(Symbol::new_variable(&padding, false));
                        }
                    }
                }
                parent.fix_depth();
                Ok(())
            }
        }
    }
    /// Same as pad, but not in-place
    #[text_signature = "($self, padding, spread, depth, /)"]
    fn create_padded(&self, padding: String, spread: u32, depth: u32) -> PyResult<PySymbol> {
        let mut new_symbol = (*self.inner).clone();
        for level in 0..depth {
            for node in new_symbol.iter_level_mut(level) {
                let nc = node.childs.len() as u32;
                for _ in nc..spread {
                    node.childs.push(Symbol::new_variable(&padding, false));
                }
            }
        }
        new_symbol.fix_depth();
        Ok(PySymbol::new(new_symbol))
    }

    #[text_signature = "($self, decorations, /)"]
    fn latex_with_deco(&self, decorations: Vec<PyDecoration>) -> PyResult<String> {
        let decorations = decorations
            .iter()
            .map(|deco| Decoration {
                path: &deco.path,
                pre: &deco.pre,
                post: &deco.post,
            })
            .collect::<Vec<_>>();
        Ok(dump_latex(&self.inner, decorations, false))
    }

    #[text_signature = "($self, decorations, /)"]
    fn latex_with_colors(&self, colors: Vec<(String, Path)>) -> PyResult<String> {
        // Storing the color strings
        let colors_code = colors
            .iter()
            .map(|(color, _)| format!("\\textcolor{{{}}}{{", color))
            .collect::<Vec<_>>();
        let decorations = colors
            .iter()
            .zip(colors_code.iter())
            .map(|((_, path), color)| Decoration {
                path,
                pre: color,
                post: "}",
            })
            .collect::<Vec<_>>();
        Ok(dump_latex(&self.inner, decorations, false))
    }
}

fn op_to_string(op: &CompareOp) -> &str {
    match op {
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}

#[pyproto]
impl PyObjectProtocol for PySymbol {
    fn __str__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.inner))
    }

    fn __hash__(&self) -> PyResult<isize> {
        let mut state = DefaultHasher::new();
        (*self.inner).hash(&mut state);
        Ok(state.finish() as isize)
    }

    fn __richcmp__(&self, other: PySymbol, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(*self.inner == *other.inner),
            CompareOp::Ne => Ok(*self.inner != *other.inner),
            _ => Err(PyErr::new::<NotImplementedError, _>(format!(
                "Comparison operator {} for Symbol is not implemented yet!",
                op_to_string(&op)
            ))),
        }
    }
    // Could replaced by dict argument in pyclass macro
    fn __setattr__(&'p mut self, name: &'p str, value: PyObject) -> PyResult<()> {
        self.try_release_attribute(name);
        self.attributes.insert(name.to_string(), value);
        Ok(())
    }

    fn __getattr__(&'p self, name: &'p str) -> PyResult<&'p PyObject> {
        self.attributes
            .get(name)
            .ok_or(PyErr::new::<KeyError, _>(format!(
                "No Attribute with key \"{}\" found!",
                name
            )))
    }

    fn __delattr__(&mut self, name: &str) -> PyResult<()> {
        self.try_release_attribute(name);
        Ok(())
    }
}

#[pyproto]
impl PyGCProtocol for PySymbol {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), pyo3::PyTraverseError> {
        for obj in self.attributes.values() {
            visit.call(obj)?
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        for obj in self.attributes.values() {
            let gil = GILGuard::acquire();
            let py = gil.python();
            py.release(obj);
        }
    }
}

impl fmt::Debug for PySymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}
