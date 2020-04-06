use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::str;

use std::collections::VecDeque;

type FlagType = u32;

mod symbol_flags {
    use super::FlagType;
    pub const FIXED: FlagType = 1;
    pub const ROOT_ONLY: FlagType = 1 << 1;
}

pub struct SymbolIter<'a> {
    stack: Vec<&'a Symbol>,
}

impl<'a> SymbolIter<'a> {
    pub fn new(parent: &'a Symbol) -> SymbolIter {
        SymbolIter {
            stack: vec![parent],
        }
    }
}

impl<'a> Iterator for SymbolIter<'a> {
    type Item = &'a Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            None => None,
            Some(current) => {
                for child in current.childs.iter() {
                    self.stack.push(child);
                }
                Some(current)
            }
        }
    }
}

pub struct SymbolAndPathIter<'a> {
    stack: Vec<(Vec<usize>, &'a Symbol)>,
}

impl<'a> SymbolAndPathIter<'a> {
    pub fn new(parent: &'a Symbol) -> SymbolAndPathIter {
        SymbolAndPathIter {
            stack: vec![(vec![], parent)],
        }
    }
}

impl<'a> Iterator for SymbolAndPathIter<'a> {
    type Item = (Vec<usize>, &'a Symbol);

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            None => None,
            Some((path, symbol)) => {
                for (i, child) in symbol.childs.iter().enumerate() {
                    self.stack.push(([&path[..], &[i]].concat(), child));
                }
                Some((path, symbol))
            }
        }
    }
}

pub struct SymbolLevelIterMut<'a> {
    pub stack: Vec<(u32, &'a mut Symbol)>,
    pub level: u32,
}

impl<'a> Iterator for SymbolLevelIterMut<'a> {
    type Item = &'a mut Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            None => None,
            Some((level, node)) => {
                if level == self.level {
                    Some(node)
                } else {
                    for child in node.childs.iter_mut().rev() {
                        self.stack.push((level + 1, child));
                    }
                    self.next()
                }
            }
        }
    }
}

pub struct SymbolLevelIter<'a> {
    pub stack: Vec<(u32, &'a Symbol)>,
    pub level: u32,
}

impl<'a> Iterator for SymbolLevelIter<'a> {
    type Item = &'a Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            None => None,
            Some((level, node)) => {
                if level == self.level {
                    Some(node)
                } else {
                    for child in node.childs.iter().rev() {
                        self.stack.push((level + 1, child));
                    }
                    self.next()
                }
            }
        }
    }
}

pub struct SymbolBfsIter<'a> {
    pub queue: VecDeque<&'a Symbol>,
}

impl<'a> SymbolBfsIter<'a> {
    pub fn new(symbol: &'a Symbol) -> SymbolBfsIter<'a> {
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back(symbol);
        SymbolBfsIter { queue }
    }
}

impl<'a> Iterator for SymbolBfsIter<'a> {
    type Item = &'a Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            None => None,
            Some(current) => {
                for child in current.childs.iter() {
                    self.queue.push_back(child);
                }
                Some(current)
            }
        }
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone, Default)]
pub struct Symbol {
    pub ident: String,
    // Empty for non operators
    pub childs: Vec<Symbol>,
    /// The depth of the hierarchal structure
    pub depth: u32,
    // Collection of flags
    pub flags: FlagType,
    pub value: Option<i64>,
}

impl Symbol {
    pub fn only_root(&self) -> bool {
        self.flags & symbol_flags::ROOT_ONLY != 0
    }

    /// For now specifies, if a variable of function is a concrete,
    /// Starting with uppercase means true, else false.
    /// commonly known constant/function or not.
    /// Specifies if this symbol should be mapped or not.
    /// Example function: "+" Example variable constant of gravitation "G"
    /// Later on it will depend on the context if function is fixed or not.
    /// For instance "G" could stand for another variable, but not the constant of gravitation.
    /// Remove this later when predicates are implemented
    #[inline]
    pub fn fixed(&self) -> bool {
        self.flags & symbol_flags::FIXED != 0
    }
    #[inline]
    pub fn operator(&self) -> bool {
        !self.childs.is_empty()
    }

    /// Assumes a spread of 2
    pub fn density(&self) -> f32 {
        let spread: i32 = 2;
        let size = self.parts().map(|_| 1).sum::<i32>();
        let max_size = (0..self.depth).map(|i| spread.pow(i)).sum::<i32>();
        size as f32 / max_size as f32
    }

    /// Returns the amount of childs
    #[inline]
    pub fn size(&self) -> u32 {
        self.childs.iter().map(|c| c.size()).sum::<u32>() + 1
    }

    fn print_tree_impl(&self, buffer: &mut String, indent: usize) {
        let ident = if self.ident.is_empty() {
            "?"
        } else {
            &self.ident
        };
        let indent_str = String::from_utf8(vec![b' '; indent]).unwrap();
        buffer.push_str(&indent_str);
        buffer.push_str(ident);
        buffer.push_str(&"\n");

        for child in self.childs.iter() {
            child.print_tree_impl(buffer, indent + 1)
        }
    }

    pub fn print_tree(&self) -> String {
        let mut code = String::new();
        self.print_tree_impl(&mut code, 0);
        code
    }

    /// Traverse depth first order
    pub fn parts(&self) -> SymbolIter {
        SymbolIter::new(self)
    }

    pub fn parts_with_path(&self) -> SymbolAndPathIter {
        SymbolAndPathIter::new(self)
    }

    pub fn parts_with_path_mut(&mut self) -> SymbolAndPathIter {
        SymbolAndPathIter::new(self)
    }

    /// Iterates all the childs of a specified level
    /// # Example
    /// ```
    /// use core::Symbol;
    ///
    /// let a = Symbol::new_variable("a", false);
    /// let b = Symbol::new_variable("b", false);
    /// let root = Symbol::new_operator("c", true, false, vec![a,b]);
    ///
    /// let actual = root.iter_level(1)
    ///     .map(|s| &s.ident)
    ///     .collect::<Vec<_>>();
    /// let expected = ["a", "b"];
    /// assert_eq!(actual, expected);
    /// ```
    pub fn iter_level(&self, level: u32) -> SymbolLevelIter {
        SymbolLevelIter {
            stack: vec![(0, self)],
            level,
        }
    }

    /// Same as iter_level but allows to mut the items
    pub fn iter_level_mut(&mut self, level: u32) -> SymbolLevelIterMut {
        SymbolLevelIterMut {
            stack: vec![(0, self)],
            level,
        }
    }

    /// Traverse all the childs in breath first order
    /// # Example
    ///```
    /// use core::Symbol;
    ///
    /// let a = Symbol::new_variable("a", false);
    /// let b = Symbol::new_variable("b", false);
    /// let root = Symbol::new_operator("c", true, false, vec![a,b]);
    ///
    /// let actual = root.iter_bfs()
    ///     .map(|s| &s.ident)
    ///     .collect::<Vec<_>>();
    /// let expected = ["c", "a", "b"];
    /// assert_eq!(actual, expected);
    ///```
    pub fn iter_bfs(&self) -> SymbolBfsIter {
        SymbolBfsIter::new(self)
    }

    pub fn embed(
        &self,
        dict: &HashMap<String, i16>,
        padding: i16,
        spread: usize,
    ) -> Result<(Vec<i16>, Vec<Vec<i16>>), String> {
        let mut ref_to_index: HashMap<&Self, i16> = HashMap::new();
        let mut embedded = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                ref_to_index.insert(s, i as i16);
                dict.get(&s.ident)
                    .and_then(|i| Some(*i))
                    .ok_or(format!("Unknown ident {}", s.ident))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let padding_index = embedded.len() as i16;
        embedded.push(padding);

        // self, ..childs, (parent later)
        let mut index_map = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                let mut row = Vec::with_capacity(spread + 2);
                row.push(i as i16);
                for child in s.childs.iter() {
                    row.push(ref_to_index[child])
                }
                while row.len() < spread + 1 {
                    row.push(padding_index);
                }
                row
            })
            .collect::<Vec<Vec<i16>>>();

        // Append parent
        // root has no parent
        index_map[0].push(padding_index);
        for parent in self.iter_bfs() {
            let parent_index = ref_to_index[parent];
            for child in parent.childs.iter() {
                let index = ref_to_index[child] as usize;
                index_map[index].push(parent_index);
            }
        }

        return Ok((embedded, index_map));
    }

    /// Returns the item at the specified path
    pub fn at<'a>(&'a self, path: &[usize]) -> Option<&'a Symbol> {
        let mut current = self;
        for i in path.iter() {
            match &current.childs.get(*i) {
                None => return None,
                Some(next) => current = next,
            }
        }
        Some(current)
    }

    pub fn set_ident_at(&mut self, path: &[usize], ident: String) -> Result<(), ()> {
        let mut current = self;
        for i in path.iter() {
            match current.childs.get_mut(*i) {
                None => return Err(()),
                Some(next) => current = next,
            }
        }
        current.ident = ident;
        Ok(())
    }

    pub fn fix_depth(&mut self) {
        for child in self.childs.iter_mut() {
            child.fix_depth();
        }
        self.depth = Symbol::calc_depth(&self.childs);
    }

    // Symbol creation
    pub fn create_flags(fixed: bool, only_root: bool) -> FlagType {
        let fixed = if fixed { symbol_flags::FIXED } else { 0 };
        let only_root = if only_root {
            symbol_flags::ROOT_ONLY
        } else {
            0
        };
        fixed | only_root
    }

    pub fn new_number(value: i64) -> Symbol {
        Symbol {
            ident: value.to_string(),
            depth: 1,
            flags: symbol_flags::FIXED,
            childs: Vec::new(),
            value: Some(value),
        }
    }
    pub fn new_variable(ident: &str, fixed: bool) -> Symbol {
        Symbol {
            ident: String::from(ident),
            depth: 1,
            flags: if fixed { symbol_flags::FIXED } else { 0 },
            childs: Vec::new(),
            value: None,
        }
    }

    pub fn new_variable_from_string(ident: String, fixed: bool) -> Symbol {
        Symbol {
            flags: if fixed { symbol_flags::FIXED } else { 0 },
            ident,
            depth: 1,
            childs: Vec::new(),
            value: None,
        }
    }

    pub fn new_operator(ident: &str, fixed: bool, only_root: bool, childs: Vec<Symbol>) -> Symbol {
        Symbol {
            ident: String::from(ident),
            depth: Symbol::calc_depth(&childs),
            flags: Symbol::create_flags(fixed, only_root),
            childs,
            value: None,
            // only_root,
        }
    }

    pub fn new_operator_from_string(
        ident: String,
        fixed: bool,
        only_root: bool,
        childs: Vec<Symbol>,
    ) -> Symbol {
        Symbol {
            ident,
            depth: Symbol::calc_depth(&childs),
            childs,
            value: None,
            flags: Symbol::create_flags(fixed, only_root),
        }
    }

    pub fn calc_depth(childs: &[Symbol]) -> u32 {
        childs.iter().map(|c| c.depth).max().unwrap_or(0) + 1
    }

    pub fn get_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::context::Context;

    #[test]
    fn calc_depth_unary_op() {
        let v = Symbol::new_variable("a", false);
        let o = Symbol::new_operator("o", true, false, vec![v]);

        assert_eq!(o.depth, 2);
    }

    #[test]
    fn calc_depth_complex_op() {
        let a = Symbol::new_variable("a", false);
        let b = Symbol::new_variable("b", false);
        let c = Symbol::new_variable("c", false);

        let o = Symbol::new_operator("o", true, false, vec![a]);
        let u = Symbol::new_operator("u", true, false, vec![o, b]);
        let v = Symbol::new_operator("v", true, false, vec![u, c]);

        assert_eq!(v.depth, 4);
    }

    #[test]
    fn parts() {
        let a = Symbol::new_variable("a", false);
        let b = Symbol::new_variable("b", false);
        let c = Symbol::new_variable("c", false);

        let o = Symbol::new_operator("o", true, false, vec![a]);
        let u = Symbol::new_operator("u", true, false, vec![o, b]);
        let v = Symbol::new_operator("v", true, false, vec![u, c]);

        let actual: Vec<_> = v.parts().map(|s| &s.ident).collect();
        let expected = vec!["v", "c", "u", "b", "o", "a"];

        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }

    #[test]
    fn parts_with_path() {
        let a = Symbol::new_variable("a", false);
        let b = Symbol::new_variable("b", false);
        let c = Symbol::new_variable("c", false);

        let o = Symbol::new_operator("o", true, false, vec![a]);
        let u = Symbol::new_operator("u", true, false, vec![o, b]);
        let v = Symbol::new_operator("v", true, false, vec![u, c]);

        assert_eq!(v.parts_with_path().count(), 6);

        for (path, symbol) in v.parts_with_path() {
            assert_eq!(v.at(&path).expect(&format!("Symbol at {:?}", path)), symbol);
        }
    }

    #[test]
    fn at_in_bound() {
        let a = Symbol::new_variable("a", false);
        let b = Symbol::new_variable("b", false);
        let c = Symbol::new_variable("c", false);

        let o = Symbol::new_operator("o", true, false, vec![a]);
        let u = Symbol::new_operator("u", true, false, vec![o, b]);
        let v = Symbol::new_operator("v", true, false, vec![u, c]);

        let actual = &v.at(&[0, 1]).expect("retuning value").ident;
        let expected = "b";

        assert_eq!(actual, expected);
    }

    #[test]
    fn at_out_of_bound() {
        let a = Symbol::new_variable("a", false);
        let b = Symbol::new_variable("b", false);
        let c = Symbol::new_variable("c", false);

        let o = Symbol::new_operator("o", true, false, vec![a]);
        let u = Symbol::new_operator("u", true, false, vec![o, b]);
        let v = Symbol::new_operator("v", true, false, vec![u, c]);

        let actual = v.at(&[0, 4]);
        let expected: Option<&Symbol> = None;

        assert_eq!(actual, expected);
    }

    fn fix_dict(dict: HashMap<&str, i16>) -> HashMap<String, i16> {
        dict.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<String, _>>()
    }

    #[test]
    fn embed_full_and_balanced() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let padding = 0;
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "*" => 3,
            "a" => 4,
            "b" => 5,
            "c" => 6,
            "d" => 7,
        };
        let dict = fix_dict(dict);
        let spread = 2;
        let (embedding, indices) = symbol.embed(&dict, padding, spread).unwrap();

        assert_eq!(embedding, vec![1, 2, 3, 4, 5, 6, 7, 0]);

        assert_eq!(indices.len(), 7);
        assert_eq!(indices[0], vec![0, 1, 2, 7]); // *=+
        assert_eq!(indices[1], vec![1, 3, 4, 0]); // a+b
        assert_eq!(indices[2], vec![2, 5, 6, 0]); // c*d
        assert_eq!(indices[3], vec![3, 7, 7, 1]); // a
        assert_eq!(indices[4], vec![4, 7, 7, 1]); // b
        assert_eq!(indices[5], vec![5, 7, 7, 2]); // c
        assert_eq!(indices[6], vec![6, 7, 7, 2]); // d
    }

    #[test]
    fn embed_not_full() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c").unwrap();
        let padding = 0;
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "a" => 3,
            "b" => 4,
            "c" => 5,
        };
        let dict = fix_dict(dict);
        let spread = 3;
        let (embedding, indices) = symbol.embed(&dict, padding, spread).unwrap();

        assert_eq!(embedding, vec![1, 2, 5, 3, 4, 0]); // =, +, c, a, b, <PAD>
        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], vec![0, 1, 2, 5, 5]); // *=c
        assert_eq!(indices[1], vec![1, 3, 4, 5, 0]); // a+b
        assert_eq!(indices[2], vec![2, 5, 5, 5, 0]); // c
        assert_eq!(indices[3], vec![3, 5, 5, 5, 1]); // a
        assert_eq!(indices[4], vec![4, 5, 5, 5, 1]); // b
    }

    #[test]
    fn size() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c").unwrap();

        assert_eq!(symbol.size(), 5);
    }
}
