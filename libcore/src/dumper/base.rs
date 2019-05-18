use crate::parser::Precedence;
use crate::Symbol;
use std::collections::{HashMap, HashSet};

pub enum FormatItem {
    Tag(&'static str),
    Child(usize),
}

// pub struct FormatContext<'a> {
//     operators: SpecialSymbols<'a>,
//     formats: SpecialFormatRules,
//     location: FormatingLocation,
// }

// impl<'a> FormatContext<'a> {
//     pub fn deeper(&self) {
//         if let self.operators
//     }
// }

pub struct FormatingLocation {
    depth: usize,
    on_track: bool,
}

impl FormatingLocation {
    pub fn new() -> FormatingLocation {
        FormatingLocation {
            depth: 0,
            on_track: true,
        }
    }
    pub fn deeper(&self, child_id: usize, path: &[usize]) -> FormatingLocation {
        FormatingLocation {
            depth: self.depth + 1,
            on_track: path.len() > self.depth && self.on_track && path[self.depth] == child_id,
        }
    }

    pub fn on_track(&self, path: &[usize]) -> bool {
        self.on_track && self.depth == path.len()
    }
}

pub struct SpecialFormatRules {
    pub symbols: HashMap<&'static str, &'static str>,
    pub functions: HashMap<&'static str, Vec<FormatItem>>,
}

pub struct SpecialSymbols<'a> {
    pub infix: HashMap<&'a str, Precedence>,
    pub postfix: HashSet<&'a str>,
    pub prefix: HashSet<&'a str>,
    pub format: SpecialFormatRules,
}

const EMPTY_VEC: &'static [usize] = &[];

impl<'a> SpecialSymbols<'a> {
    pub fn get(&self, key: &'static str) -> &'static str {
        self.format.symbols.get(key).unwrap_or(&key)
    }

    pub fn format_function(
        &self,
        decoration: &Option<Decoration>,
        location: &FormatingLocation,
        symbol: &Symbol,
        mut code: &mut String,
    ) -> Option<()> {
        match self.format.functions.get::<str>(&symbol.ident) {
            Some(rules) => {
                for rule in rules.iter() {
                    match rule {
                        FormatItem::Tag(tag) => code.push_str(tag),
                        FormatItem::Child(index) => {
                            let path = if let Some(deco) = decoration {
                                deco.path
                            } else {
                                &EMPTY_VEC
                            };
                            dump_base(
                                self,
                                symbol.childs.get(*index).expect(""),
                                &mut code,
                                decoration,
                                location.deeper(*index, &path),
                            )
                        }
                    }
                }
                Some(())
            }
            None => None,
        }
    }
}

const P_HIGHEST: Precedence = Precedence::PFaculty;
fn get_precedence_or_default<'a>(
    special_symbols: &'a SpecialSymbols,
    ident: &str,
) -> &'a Precedence {
    match special_symbols.infix.get(&ident[..]) {
        None => &P_HIGHEST,
        Some(pre) => pre,
    }
}

fn dump_atomic(
    special_symbols: &SpecialSymbols,
    symbol: &Symbol,
    bracket: bool,
    string: &mut String,
    decoration: &Option<Decoration>,
    location: FormatingLocation,
) {
    if bracket {
        string.push_str(special_symbols.get("("));
        dump_base(special_symbols, symbol, string, decoration, location);
        string.push_str(special_symbols.get(")"));
    } else {
        dump_base(special_symbols, symbol, string, decoration, location);
    }
}

pub struct Decoration<'a> {
    pub path: &'a [usize],
    pub pre: &'static str,
    pub post: &'static str,
}

/// Improvement hint: implement a version taking a Writer instead a String
pub fn dump_base(
    special_symbols: &SpecialSymbols,
    symbol: &Symbol,
    mut string: &mut String,
    decoration: &Option<Decoration>,
    location: FormatingLocation,
) {
    let path = if let Some(deco) = decoration {
        deco.path
    } else {
        &EMPTY_VEC
    };
    let decoration = if location.on_track(path) {
        decoration
    } else {
        &None
    };
    if let Some(decoration) = decoration {
        string.push_str(decoration.pre);
    }

    match symbol.childs.len() {
        0 => string.push_str(&symbol.ident),
        1 if special_symbols.postfix.contains(&symbol.ident[..]) => {
            let child = &symbol.childs[0];
            let pre_child = get_precedence_or_default(special_symbols, &child.ident);
            dump_atomic(
                special_symbols,
                child,
                pre_child < &P_HIGHEST,
                string,
                decoration,
                location,
            );
            string.push_str(&symbol.ident);
        }
        1 if special_symbols.prefix.contains(&symbol.ident[..]) => {
            match special_symbols.format_function(decoration, &location, &symbol, &mut string) {
                Some(_) => (),
                None => {
                    string.push_str(&symbol.ident);
                    let child = &symbol.childs[0];
                    let pre_child = get_precedence_or_default(special_symbols, &child.ident);
                    dump_atomic(
                        special_symbols,
                        child,
                        pre_child < &P_HIGHEST,
                        string,
                        decoration,
                        location.deeper(0, path),
                    );
                }
            }
        }
        2 if special_symbols.infix.contains_key(&symbol.ident[..]) => {
            match special_symbols.format_function(decoration, &location, &symbol, &mut string) {
                Some(_) => (),
                None => {
                    let pre_root = get_precedence_or_default(special_symbols, &symbol.ident);
                    let left = &symbol.childs[0];
                    let right = &symbol.childs[1];
                    let pre_left = get_precedence_or_default(special_symbols, &left.ident);
                    let pre_right = get_precedence_or_default(special_symbols, &right.ident);
                    dump_atomic(
                        special_symbols,
                        left,
                        pre_left < pre_root,
                        string,
                        decoration,
                        location.deeper(0, path),
                    );
                    string.push_str(&symbol.ident);
                    dump_atomic(
                        special_symbols,
                        right,
                        pre_right < pre_root,
                        string,
                        decoration,
                        location.deeper(1, path),
                    );
                }
            }
        }
        _ => {
            string.push_str(&symbol.ident);
            let mut first = true;
            string.push_str(special_symbols.get("("));
            for (i, child) in symbol.childs.iter().enumerate() {
                if !first {
                    string.push_str(", ");
                }
                dump_base(
                    special_symbols,
                    child,
                    string,
                    decoration,
                    location.deeper(i, path),
                );
                first = false;
            }
            string.push_str(special_symbols.get(")"));
        }
    };

    if let Some(decoration) = decoration {
        string.push_str(decoration.post);
    }
}
