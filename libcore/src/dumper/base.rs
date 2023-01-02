use crate::common::RefEquality;
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
    /// position, discard brackets
    Child(usize, bool),
}

pub struct FormatContext<'a> {
    pub operators: Operators<'a>,
    pub formats: SpecialFormatRules,
    pub decoration: &'a [Decoration<'a>],
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

pub trait DumpingHook<'a> {
    fn pre(&mut self, symbol: &'a Symbol, position: usize);
    fn post(&mut self, symbol: &'a Symbol, position: usize);
}

pub struct NoOpDumpingHood {}

impl<'a> DumpingHook<'a> for NoOpDumpingHood {
    fn pre(&mut self, _symbol: &Symbol, _position: usize) {}
    fn post(&mut self, _symbol: &Symbol, _position: usize) {}
}

#[derive(Default)]
pub struct RecorderDumpingHook<'a> {
    pub begin_positions: HashMap<RefEquality<'a, Symbol>, usize>,
    pub end_positions: HashMap<RefEquality<'a, Symbol>, usize>,
}

impl<'a> DumpingHook<'a> for RecorderDumpingHook<'a> {
    fn pre(&mut self, symbol: &'a Symbol, position: usize) {
        self.begin_positions.insert(RefEquality(symbol), position);
    }
    fn post(&mut self, symbol: &'a Symbol, position: usize) {
        self.end_positions.insert(RefEquality(symbol), position);
    }
}

const P_HIGHEST: Precedence = Precedence::PFaculty;
impl<'a> FormatContext<'a> {
    pub fn get<'b>(&self, key: &'b str) -> &'b str {
        self.formats.symbols.get(key).unwrap_or(&key)
    }

    pub fn format_function(
        &'a self,
        symbol: &'a Symbol,
        location: &FormattingLocation,
        make_unary_minus: bool,
        hook: &mut dyn DumpingHook<'a>,
        mut code: &mut String,
    ) -> Option<()> {
        self.formats
            .functions
            .get::<str>(&symbol.ident)
            .and_then(|rules| {
                for rule in rules.iter() {
                    match rule {
                        FormatItem::Tag(tag) => code.push_str(tag),
                        FormatItem::Child(index, no_brackets) => {
                            let child = symbol.childs.get(*index).expect("");
                            let bracket = !*no_brackets
                                && child.depth > 1
                                && !self.formats.functions.contains_key(&child.ident[..]);
                            dump_atomic(
                                self,
                                child,
                                bracket,
                                location.deeper(*index),
                                make_unary_minus,
                                hook,
                                &mut code,
                            );
                        }
                    }
                }
                Some(())
            })
    }
    pub fn get_precedence_or_default(&self, symbol: &Symbol) -> &Precedence {
        match self.operators.infix.get(&symbol.ident[..]) {
            None => &P_HIGHEST,
            Some(pre) => pre,
        }
    }
}

fn dump_atomic<'a>(
    context: &'a FormatContext,
    symbol: &'a Symbol,
    bracket: bool,
    location: FormattingLocation,
    make_unary_minus: bool,
    hook: &mut dyn DumpingHook<'a>,
    string: &mut String,
) {
    if bracket {
        string.push_str(context.get("("));
        dump_base(context, symbol, location, make_unary_minus, hook, string);
        string.push_str(context.get(")"));
    } else {
        dump_base(context, symbol, location, make_unary_minus, hook, string);
    }
}

fn push_ident<'a>(
    context: &FormatContext,
    hook: &mut dyn DumpingHook<'a>,
    string: &mut String,
    symbol: &'a Symbol,
) {
    hook.pre(symbol, string.len());
    string.push_str(context.get(&symbol.ident));
    hook.post(symbol, string.len());
}

/// Improvement hint: implement a version taking a Writer instead a String
pub fn dump_base<'a>(
    context: &'a FormatContext,
    symbol: &'a Symbol,
    location: FormattingLocation,
    make_unary_minus: bool,
    hook: &mut dyn DumpingHook<'a>,
    mut string: &mut String,
) {
    let actual_decoration = location.select_decoration(&context.decoration);
    if let Some(decoration) = actual_decoration {
        string.push_str(decoration.pre);
    }

    if context
        .format_function(&symbol, &location, make_unary_minus, hook, &mut string)
        .is_none()
    {
        match symbol.childs.len() {
            // 0 => string.push_str(context.get(&symbol.ident)),
            0 => push_ident(context, hook, string, symbol),
            1 if context.operators.postfix.contains(&symbol.ident[..]) => {
                let child = &symbol.childs[0];
                let pre_child = context.get_precedence_or_default(&child);
                dump_atomic(
                    context,
                    child,
                    pre_child < &P_HIGHEST,
                    location,
                    make_unary_minus,
                    hook,
                    string,
                );
                push_ident(context, hook, string, symbol);
            }
            1 if context.operators.prefix.contains(&symbol.ident[..]) => {
                let child = &symbol.childs[0];
                let pre_child = context.get_precedence_or_default(&child);

                push_ident(context, hook, string, symbol);
                dump_atomic(
                    context,
                    child,
                    pre_child < &P_HIGHEST,
                    location.deeper(0),
                    make_unary_minus,
                    hook,
                    string,
                );
            }
            2 if context.operators.infix.contains_key(&symbol.ident[..]) => {
                let pre_root = context.get_precedence_or_default(&symbol);
                let left = &symbol.childs[0];
                let right = &symbol.childs[1];
                let pre_left = context.get_precedence_or_default(&left);
                let pre_right = context.get_precedence_or_default(&right);
                // Dump -1*a as -a
                // TODO: Support -(a*b) as well
                if make_unary_minus
                    && &symbol.ident == "*"
                    && pre_right == &P_HIGHEST
                    && left.value.unwrap_or_default() == -1
                {
                    string.push('-');
                    string.push_str(context.get(&right.ident));
                } else {
                    dump_atomic(
                        context,
                        left,
                        pre_left < pre_root,
                        location.deeper(0),
                        make_unary_minus,
                        hook,
                        string,
                    );
                    push_ident(context, hook, string, symbol);
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
                        make_unary_minus,
                        hook,
                        string,
                    );
                }
            }
            _ => {
                push_ident(context, hook, string, symbol);
                let mut first = true;
                string.push_str(context.get("("));
                for (i, child) in symbol.childs.iter().enumerate() {
                    if !first {
                        string.push_str(", ");
                    }
                    dump_base(
                        context,
                        child,
                        location.deeper(i),
                        make_unary_minus,
                        hook,
                        string,
                    );
                    first = false;
                }
                string.push_str(context.get(")"));
            }
        };
    }

    if let Some(decoration) = actual_decoration {
        string.push_str(decoration.post);
    }
}
