use crate::parser::Precedence;
use crate::Symbol;
use std::collections::{HashMap, HashSet};

pub enum FormatItem {
    Tag(&'static str),
    Child(usize),
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

impl<'a> SpecialSymbols<'a> {
    pub fn get(&self, key: &'static str) -> &'static str {
        self.format.symbols.get(key).unwrap_or(&key)
    }

    pub fn format_function(&self, symbol: &Symbol, mut code: &mut String) -> Option<()> {
        match self.format.functions.get::<str>(&symbol.ident) {
            Some(rules) => {
                for rule in rules.iter() {
                    match rule {
                        FormatItem::Tag(tag) => code.push_str(tag),
                        FormatItem::Child(index) => {
                            dump_base(self, symbol.childs.get(*index).expect(""), &mut code)
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
) {
    if bracket {
        string.push_str(special_symbols.get("("));
        dump_base(special_symbols, symbol, string);
        string.push_str(special_symbols.get(")"));
    } else {
        dump_base(special_symbols, symbol, string);
    }
}

pub fn dump_base(special_symbols: &SpecialSymbols, symbol: &Symbol, mut string: &mut String) {
    match symbol.childs.len() {
        0 => string.push_str(&symbol.ident),
        1 if special_symbols.postfix.contains(&symbol.ident[..]) => {
            let child = &symbol.childs[0];
            let pre_child = get_precedence_or_default(special_symbols, &child.ident);
            dump_atomic(special_symbols, child, pre_child < &P_HIGHEST, string);
            string.push_str(&symbol.ident);
        }
        1 if special_symbols.prefix.contains(&symbol.ident[..]) => {
            match special_symbols.format_function(&symbol, &mut string) {
                Some(_) => (),
                None => {
                    string.push_str(&symbol.ident);
                    let child = &symbol.childs[0];
                    let pre_child = get_precedence_or_default(special_symbols, &child.ident);
                    dump_atomic(special_symbols, child, pre_child < &P_HIGHEST, string);
                }
            }
        }
        2 if special_symbols.infix.contains_key(&symbol.ident[..]) => {
            match special_symbols.format_function(&symbol, &mut string) {
                Some(_) => (),
                None => {
                    let pre_root = get_precedence_or_default(special_symbols, &symbol.ident);
                    let left = &symbol.childs[0];
                    let right = &symbol.childs[1];
                    let pre_left = get_precedence_or_default(special_symbols, &left.ident);
                    let pre_right = get_precedence_or_default(special_symbols, &right.ident);
                    dump_atomic(special_symbols, left, pre_left < pre_root, string);
                    string.push_str(&symbol.ident);
                    dump_atomic(special_symbols, right, pre_right < pre_root, string);
                }
            }
        }
        _ => {
            string.push_str(&symbol.ident);
            let mut first = true;
            string.push_str(special_symbols.get("("));
            for child in symbol.childs.iter() {
                if !first {
                    string.push_str(", ");
                }
                dump_base(special_symbols, child, string);
                first = false;
            }
            string.push_str(special_symbols.get(")"));
        }
    };
}
