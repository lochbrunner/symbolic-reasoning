use std::str;

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

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
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
    pub fn fixed(&self) -> bool {
        self.flags & symbol_flags::FIXED != 0
    }

    pub fn parts(&self) -> SymbolIter {
        SymbolIter::new(self)
    }

    pub fn parts_with_path(&self) -> SymbolAndPathIter {
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
}

#[cfg(test)]
mod specs {
    use super::*;

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
}
