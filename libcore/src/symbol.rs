use crate::common::RefEquality;
use crate::io::bag::FitInfo;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem;
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

/// Iter with backback
pub struct SymbolBfsBackPackIter<'a, T, C> {
    // type Pack = (&'a Symbol, T);
    queue: VecDeque<(&'a Symbol, T)>,
    context: C,
    /// Generator would be better but
    packer: fn(parent: &(&'a Symbol, T), &C) -> Vec<(&'a Symbol, T)>,
}

impl<'a, T, C> SymbolBfsBackPackIter<'a, T, C> {
    pub fn new(
        symbol: &'a Symbol,
        init: T,
        context: C,
        packer: fn(parent: &(&'a Symbol, T), &C) -> Vec<(&'a Symbol, T)>,
    ) -> SymbolBfsBackPackIter<'a, T, C> {
        let mut queue = VecDeque::with_capacity(1);
        queue.push_back((symbol, init));
        Self {
            queue,
            packer,
            context,
        }
    }
}

impl<'a, T, C> Iterator for SymbolBfsBackPackIter<'a, T, C> {
    type Item = (&'a Symbol, T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            None => None,
            Some(current) => {
                for child in (self.packer)(&current, &self.context).into_iter() {
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
    // If this symbol is a number (only integers are supported yet)
    pub value: Option<i64>,
}

fn one_encode(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

pub struct Embedding {
    /// each items is a vector
    /// [ident, is_operator, is_fixed, is_number]
    pub embedded: Vec<Vec<i64>>,
    pub index_map: Option<Vec<Vec<i16>>>,
    pub positional_encoding: Option<Vec<Vec<i64>>>,
    pub label: Vec<i64>,
    pub policy: Vec<f32>,
    pub value: i64,
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
    #[inline]
    pub fn is_number(&self) -> bool {
        self.value.is_some()
    }

    /// Assumes a spread of 2
    pub fn max_spread(&self) -> u32 {
        2
    }

    pub fn density(&self) -> f32 {
        let spread = self.max_spread() as i32;
        let size = self.parts().map(|_| 1).sum::<i32>();
        let max_size = (0..self.depth).map(|i| spread.pow(i)).sum::<i32>();
        size as f32 / max_size as f32
    }

    /// Returns the amount of childs
    #[inline]
    pub fn size(&self) -> u32 {
        self.childs.iter().map(|c| c.size()).sum::<u32>() + 1
    }

    pub fn memory_usage(&self) -> usize {
        (mem::size_of::<Self>() * (1 + self.childs.capacity() - self.childs.len()))
            + self.childs.iter().map(|c| c.memory_usage()).sum::<usize>()
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

    /// Traverse depth first order
    pub fn iter_df(&self) -> SymbolIter {
        SymbolIter::new(self)
    }

    pub fn parts_with_path(&self) -> SymbolAndPathIter {
        SymbolAndPathIter::new(self)
    }

    pub fn iter_dfs_path(&self) -> SymbolAndPathIter {
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

    pub fn iter_bfs_backpack<'a, T, C>(
        &'a self,
        init: T,
        context: C,
        packer: fn(parent: &(&'a Symbol, T), &C) -> Vec<(&'a Symbol, T)>,
    ) -> SymbolBfsBackPackIter<'a, T, C> {
        SymbolBfsBackPackIter::new(self, init, context, packer)
    }

    /// Returns the number of embedded properties
    pub fn number_of_embedded_properties() -> u32 {
        3
    }

    /// Needed for transformer based architecture
    /// Using the path as digits for a representative location numbering
    /// Desired properties:
    /// * `p_a - p_b` is a metric for the relative sub-graph between node `a` and `b`
    /// * `p_a - p_b` is independent of the absolute location in the graph. Only possible with vector.
    ///     Each item per row is from an other start level.
    /// * `0` indicates undefined
    fn positional_encoding(&self, spread: usize, max_depth: u32) -> Vec<Vec<i64>> {
        struct Pack {
            offset: i64,
            // scale: i64,
            depth: u32,
        }
        struct Context {
            spread: i64,
        }

        assert_eq!(spread, 2);

        (2..(max_depth + 1))
            .rev()
            .map(|depth| {
                self.iter_bfs_backpack(
                    Pack {
                        offset: spread.pow(depth) as i64 / 2,
                        depth: if depth > 0 { depth - 1 } else { 0 },
                    },
                    Context {
                        spread: spread as i64,
                    },
                    |(parent, pack), context| {
                        parent
                            .childs
                            .iter()
                            .enumerate()
                            .map(|(i, child)| {
                                let root = if pack.depth > 0 {
                                    context.spread.pow(pack.depth - 1) * (2 * (i as i64) - 1)
                                        + pack.offset
                                } else {
                                    0
                                };
                                (
                                    child,
                                    Pack {
                                        offset: root,
                                        depth: if pack.depth > 0 { pack.depth - 1 } else { 0 },
                                    },
                                )
                            })
                            .collect()
                    },
                )
                .map(|(_, pack)| pack.offset)
                .collect()
            })
            .collect()

        // vec![buffer]
    }

    /// Needed for CNN based architecture
    /// self, ..childs, parent
    fn index_map<'a>(
        &'a self,
        spread: usize,
        padding_index: i16,
        ref_to_index: &HashMap<RefEquality<'a, Self>, i16>,
    ) -> Vec<Vec<i16>> {
        let mut index_map = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                let mut row = Vec::with_capacity(spread + 2);
                row.push(i as i16);
                for child in s.childs.iter() {
                    row.push(ref_to_index[&RefEquality(child)]);
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
            let parent_index = ref_to_index[&RefEquality(parent)];
            for child in parent.childs.iter() {
                let index = ref_to_index[&RefEquality(child)] as usize;
                index_map[index].push(parent_index);
            }
        }
        index_map.push(vec![padding_index; spread + 2]);
        index_map
    }

    /// Embeds the ident and the props (operator, fixed, number, policy)
    /// Should maybe moved to other location?
    /// If there are multiple fits per path, the last will win.
    pub fn embed(
        &self,
        dict: &HashMap<String, i16>,
        padding: i16,
        spread: usize,
        max_depth: u32,
        fits: &[FitInfo],
        useful: bool,
        index_map: bool,
        positional_encoding: bool,
    ) -> Result<Embedding, String> {
        let mut ref_to_index: HashMap<RefEquality<Self>, i16> = HashMap::new();
        let mut embedded = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                ref_to_index.insert(RefEquality(s), i as i16);
                dict.get(&s.ident)
                    .map(|i| {
                        vec![
                            *i as i64,
                            one_encode(s.operator()),
                            one_encode(s.fixed()),
                            one_encode(s.is_number()),
                        ]
                    })
                    .ok_or(format!("Unknown ident {}", s.ident))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let padding_index = embedded.len() as i16;
        embedded.push(vec![padding as i64, 0, 0, 0]);

        let index_map = if index_map {
            Some(self.index_map(spread, padding_index, &ref_to_index))
        } else {
            None
        };
        let positional_encoding = if positional_encoding {
            Some(self.positional_encoding(spread, max_depth))
        } else {
            None
        };

        // Compute label
        let mut label = vec![0; embedded.len()];
        let mut policy = vec![0.0; embedded.len()];
        for fit in fits.iter() {
            let child = self
                .at(&fit.path)
                .ok_or(format!("Symbol {} has no element at {:?}", self, fit.path))?;
            let index = ref_to_index[&RefEquality(child)] as usize;
            label[index] = fit.rule_id as i64;
            policy[index] = fit.policy.value();
        }

        Ok(Embedding {
            embedded,
            index_map,
            label,
            policy,
            positional_encoding,
            value: if useful { 1 } else { 0 },
        })
    }

    /// Replace all occurrences with predicate
    pub fn replace<F, E>(&self, replacer: &F) -> Result<Self, E>
    where
        F: Fn(&Symbol) -> Result<Option<Symbol>, E>,
    {
        let mut root = replacer(&self)?.unwrap_or_else(|| self.clone());
        root.childs = root
            .childs
            .into_iter()
            .map(|child| child.replace(replacer))
            .collect::<Result<_, E>>()?;
        Ok(root)
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
    use crate::context::{Context, Declaration};
    use crate::io::bag::Policy;

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
    fn index_map_bug_1() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b+1=(a−b)*x").unwrap();
        let ref_to_index: HashMap<RefEquality<Symbol>, i16> = symbol
            .iter_bfs()
            .enumerate()
            .map(|(i, symbol)| (RefEquality(symbol), i as i16))
            .collect();
        let index_map = symbol.index_map(2, symbol.size() as i16, &ref_to_index);
        let expected = vec![
            vec![0, 1, 2, 11],
            vec![1, 3, 4, 0],
            vec![2, 5, 6, 0],
            vec![3, 7, 8, 1],
            vec![4, 11, 11, 1],
            vec![5, 9, 10, 2],
            vec![6, 11, 11, 2],
            vec![7, 11, 11, 3],
            vec![8, 11, 11, 3],
            vec![9, 11, 11, 5],
            vec![10, 11, 11, 5],
            vec![11, 11, 11, 11],
        ];
        assert_eq!(index_map, expected);
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
        let Embedding {
            embedded,
            index_map,
            value,
            ..
        } = symbol
            .embed(
                &dict,
                padding,
                spread,
                symbol.depth,
                &vec![],
                true,
                true,
                false,
            )
            .unwrap();
        let embedded = embedded.iter().map(|emb| emb[0]).collect::<Vec<i64>>();

        assert_eq!(embedded, vec![1, 2, 3, 4, 5, 6, 7, 0]);

        assert!(index_map.is_some());
        let index_map = index_map.unwrap();
        assert_eq!(index_map.len(), 8);
        assert_eq!(index_map[0], vec![0, 1, 2, 7]); // *=+
        assert_eq!(index_map[1], vec![1, 3, 4, 0]); // a+b
        assert_eq!(index_map[2], vec![2, 5, 6, 0]); // c*d
        assert_eq!(index_map[3], vec![3, 7, 7, 1]); // a
        assert_eq!(index_map[4], vec![4, 7, 7, 1]); // b
        assert_eq!(index_map[5], vec![5, 7, 7, 2]); // c
        assert_eq!(index_map[6], vec![6, 7, 7, 2]); // d
        assert_eq!(index_map[7], vec![7, 7, 7, 7]); // d

        assert_eq!(value, 1);
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
        let Embedding {
            embedded,
            index_map,
            ..
        } = symbol
            .embed(
                &dict,
                padding,
                spread,
                symbol.depth,
                &vec![],
                true,
                true,
                false,
            )
            .unwrap();
        let embedded = embedded.iter().map(|emb| emb[0]).collect::<Vec<i64>>();
        assert!(index_map.is_some());
        let index_map = index_map.unwrap();
        assert_eq!(embedded, vec![1, 2, 5, 3, 4, 0]); // =, +, c, a, b, <PAD>
        assert_eq!(index_map.len(), 6);
        assert_eq!(index_map[0], vec![0, 1, 2, 5, 5]); // *=c
        assert_eq!(index_map[1], vec![1, 3, 4, 5, 0]); // a+b
        assert_eq!(index_map[2], vec![2, 5, 5, 5, 0]); // c
        assert_eq!(index_map[3], vec![3, 5, 5, 5, 1]); // a
        assert_eq!(index_map[4], vec![4, 5, 5, 5, 1]); // b
        assert_eq!(index_map[5], vec![5, 5, 5, 5, 5]); // padding
    }

    #[test]
    fn embed_labels() {
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
        let Embedding { label, .. } = symbol
            .embed(
                &dict,
                padding,
                spread,
                symbol.depth,
                &vec![
                    FitInfo {
                        rule_id: 1,
                        path: vec![0, 0],
                        policy: Policy::Positive,
                    },
                    FitInfo {
                        rule_id: 2,
                        path: vec![0, 1],
                        policy: Policy::Positive,
                    },
                ],
                true,
                true,
                false,
            )
            .unwrap();

        assert_eq!(label, vec![0, 0, 0, 1, 2, 0, 0, 0,]);
    }

    #[test]
    fn positional_encoding_full() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let spread = 2;
        // = 4 2
        // + 2 1
        // * 6 3
        // a 1 0
        // b 3 0
        // c 5 0
        // d 7 0

        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        let expected = vec![vec![4, 2, 6, 1, 3, 5, 7], vec![2, 1, 3, 0, 0, 0, 0]];
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn positional_encoding_simple() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c").unwrap();

        let spread = 2;
        // = 4 2
        // + 2 1
        // c 6 3
        // a 1 0
        // b 3 0
        let expected = vec![vec![4, 2, 6, 1, 3], vec![2, 1, 3, 0, 0]];
        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn positional_encoding_deep() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+(b-d)=c").unwrap();
        let spread = 2;
        // =  8  4  2
        // +  4  2  1
        // c 12  6  3
        // a  2  1  0
        // -  6  3  0
        // b  5  0  0
        // d  7  0  0

        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        let expected = vec![
            vec![8, 4, 12, 2, 6, 5, 7],
            vec![4, 2, 6, 1, 3, 0, 0],
            vec![2, 1, 3, 0, 0, 0, 0],
        ];
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn size() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c").unwrap();

        assert_eq!(symbol.size(), 5);
    }

    #[test]
    fn replace_term() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c").unwrap();
        fn replacer(part: &Symbol) -> Result<Option<Symbol>, ()> {
            if part.ident == "a" {
                Ok(Some(Symbol::new_variable("d", false)))
            } else {
                Ok(None)
            }
        }
        let actual = symbol.replace(&replacer).unwrap();
        let expected = Symbol::parse(&context, "d+b=c").unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn replace_function() {
        let mut context = Context::standard();
        context
            .declarations
            .insert("sqrt".to_string(), Declaration::function(true));
        context
            .declarations
            .insert("root".to_string(), Declaration::function(true));

        let symbol = Symbol::parse(&context, "sqrt(a+b)=c").unwrap();
        fn replacer(part: &Symbol) -> Result<Option<Symbol>, ()> {
            if part.ident == "sqrt" {
                let mut childs = part.childs.clone();
                childs.push(Symbol::new_number(2));
                Ok(Some(Symbol::new_operator("root", true, false, childs)))
            } else {
                Ok(None)
            }
        }
        let actual = symbol.replace(&replacer).unwrap();
        let expected = Symbol::parse(&context, "root(a+b, 2)=c").unwrap();
        assert_eq!(actual, expected);
    }
}
