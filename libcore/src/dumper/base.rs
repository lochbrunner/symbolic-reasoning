use crate::parser::Precedence;
use crate::Symbol;
use std::collections::{HashMap, HashSet};

pub struct FormattingLocation {
    path: Vec<usize>,
}

impl FormattingLocation {
    pub fn new() -> FormattingLocation {
        FormattingLocation { path: vec![] }
    }
    /// Appends the child id and returns a new location
    pub fn deeper(&self, child_id: usize) -> FormattingLocation {
        FormattingLocation {
            path: [&self.path[..], &[child_id]].concat(),
        }
    }

    pub fn select_decoration<'a>(
        &self,
        decorations: &'a [Decoration<'a>],
    ) -> Option<&'a Decoration<'a>> {
        decorations.iter().find(|deco| {
            deco.path.len() == self.path.len()
                && deco.path.iter().zip(&self.path).all(|(a, b)| a == b)
        })
    }
}

pub enum FormatItem {
    Tag(&'static str),
    Child(usize),
}

pub struct FormatContext<'a> {
    pub operators: Operators<'a>,
    pub formats: SpecialFormatRules,
    pub decoration: Vec<Decoration<'a>>,
}

impl<'a> FormatContext<'a> {}

pub struct SpecialFormatRules {
    pub symbols: HashMap<&'static str, &'static str>,
    pub functions: HashMap<&'static str, Vec<FormatItem>>,
}

pub struct Operators<'a> {
    pub infix: HashMap<&'a str, Precedence>,
    pub postfix: HashSet<&'a str>,
    pub prefix: HashSet<&'a str>,
    pub non_associative: HashSet<&'a str>,
}

pub struct Decoration<'a> {
    pub path: &'a [usize],
    pub pre: &'a str,
    pub post: &'a str,
}

const P_HIGHEST: Precedence = Precedence::PFaculty;
impl<'a> FormatContext<'a> {
    pub fn get<'b>(&self, key: &'b str) -> &'b str {
        self.formats.symbols.get(key).unwrap_or(&key)
    }

    pub fn format_function(
        &self,
        symbol: &Symbol,
        location: &FormattingLocation,
        mut code: &mut String,
    ) -> Option<()> {
        match self.formats.functions.get::<str>(&symbol.ident) {
            Some(rules) => {
                for rule in rules.iter() {
                    match rule {
                        FormatItem::Tag(tag) => code.push_str(tag),
                        FormatItem::Child(index) => {
                            let child = symbol.childs.get(*index).expect("");
                            let bracket = child.depth > 1
                                && !self.formats.functions.contains_key(&child.ident[..]);
                            dump_atomic(self, child, bracket, location.deeper(*index), &mut code);
                        }
                    }
                }
                Some(())
            }
            None => None,
        }
    }
    pub fn get_precedence_or_default(&self, symbol: &Symbol) -> &Precedence {
        match self.operators.infix.get(&symbol.ident[..]) {
            None => &P_HIGHEST,
            Some(pre) => pre,
        }
    }
}

fn dump_atomic(
    context: &FormatContext,
    symbol: &Symbol,
    bracket: bool,
    location: FormattingLocation,
    string: &mut String,
) {
    if bracket {
        string.push_str(context.get("("));
        dump_base(context, symbol, location, string);
        string.push_str(context.get(")"));
    } else {
        dump_base(context, symbol, location, string);
    }
}

/// Improvement hint: implement a version taking a Writer instead a String
pub fn dump_base(
    context: &FormatContext,
    symbol: &Symbol,
    location: FormattingLocation,
    mut string: &mut String,
) {
    let actual_decoration = location.select_decoration(&context.decoration);
    if let Some(decoration) = actual_decoration {
        string.push_str(decoration.pre);
    }

    match symbol.childs.len() {
        0 => string.push_str(context.get(&symbol.ident)),
        1 if context.operators.postfix.contains(&symbol.ident[..]) => {
            let child = &symbol.childs[0];
            let pre_child = context.get_precedence_or_default(&child);
            dump_atomic(context, child, pre_child < &P_HIGHEST, location, string);
            string.push_str(context.get(&symbol.ident));
        }
        1 if context.operators.prefix.contains(&symbol.ident[..]) => {
            match context.format_function(&symbol, &location, &mut string) {
                Some(_) => (),
                None => {
                    string.push_str(context.get(&symbol.ident));
                    let child = &symbol.childs[0];
                    let pre_child = context.get_precedence_or_default(&child);
                    dump_atomic(
                        context,
                        child,
                        pre_child < &P_HIGHEST,
                        location.deeper(0),
                        string,
                    );
                }
            }
        }
        2 if context.operators.infix.contains_key(&symbol.ident[..]) => {
            match context.format_function(&symbol, &location, &mut string) {
                Some(_) => (),
                None => {
                    let pre_root = context.get_precedence_or_default(&symbol);
                    let left = &symbol.childs[0];
                    let right = &symbol.childs[1];
                    let pre_left = context.get_precedence_or_default(&left);
                    let pre_right = context.get_precedence_or_default(&right);
                    dump_atomic(
                        context,
                        left,
                        pre_left < pre_root,
                        location.deeper(0),
                        string,
                    );
                    string.push_str(context.get(&symbol.ident));

                    dump_atomic(
                        context,
                        right,
                        pre_right < pre_root
                            || (pre_right == pre_root
                                && context
                                    .operators
                                    .non_associative
                                    .contains(&symbol.ident[..])),
                        location.deeper(1),
                        string,
                    );
                }
            }
        }
        _ => {
            string.push_str(context.get(&symbol.ident));
            let mut first = true;
            string.push_str(context.get("("));
            for (i, child) in symbol.childs.iter().enumerate() {
                if !first {
                    string.push_str(", ");
                }
                dump_base(context, child, location.deeper(i), string);
                first = false;
            }
            string.push_str(context.get(")"));
        }
    };

    if let Some(decoration) = actual_decoration {
        string.push_str(decoration.post);
    }
}
