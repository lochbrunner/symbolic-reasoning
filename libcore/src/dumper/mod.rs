use crate::parser::Precedence; // TODO: Use local precedence table here
use crate::symbol::Symbol;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

struct SpecialSymbols<'a> {
    pub infix: HashMap<&'a str, Precedence>,
    pub postfix: HashSet<&'a str>,
}

const P_HIGHEST: Precedence = Precedence::PFaculty;
fn get_precedence_or_default<'a>(
    special_symbols: &'a SpecialSymbols,
    ident: &String,
) -> &'a Precedence {
    match special_symbols.infix.get(&ident[..]) {
        None => &P_HIGHEST,
        Some(pre) => pre,
    }
}

fn dump_impl(special_symbols: &SpecialSymbols, symbol: &Symbol, string: &mut String) {
    if !symbol.childs.is_empty() {
        if special_symbols.infix.contains_key(&symbol.ident[..]) {
            let pre_root = get_precedence_or_default(special_symbols, &symbol.ident);

            assert_eq!(symbol.childs.len(), 2);
            let left = &symbol.childs[0];
            let right = &symbol.childs[1];
            let pre_left = get_precedence_or_default(special_symbols, &left.ident);
            let pre_right = get_precedence_or_default(special_symbols, &right.ident);
            if pre_left < pre_root {
                string.push_str("(");
                dump_impl(special_symbols, left, string);
                string.push_str(")");
            } else {
                dump_impl(special_symbols, left, string);
            }
            string.push_str(&symbol.ident);
            if pre_right < pre_root {
                string.push_str("(");
                dump_impl(special_symbols, right, string);
                string.push_str(")");
            } else {
                dump_impl(special_symbols, right, string);
            }
        } else if special_symbols.postfix.contains(&symbol.ident[..]) {
            assert_eq!(symbol.childs.len(), 1);
            let child = &symbol.childs[0];
            let pre_child = get_precedence_or_default(special_symbols, &child.ident);
            if pre_child < &P_HIGHEST {
                string.push_str("(");
                dump_impl(special_symbols, child, string);
                string.push_str(")");
            } else {
                dump_impl(special_symbols, child, string);
            }
            string.push_str(&symbol.ident);
        } else {
            string.push_str(&symbol.ident);
            let mut first = true;
            string.push_str("(");
            for child in symbol.childs.iter() {
                if !first {
                    string.push_str(", ");
                }
                dump_impl(special_symbols, child, string);
                first = false;
            }
            string.push_str(")");
        }
    } else {
        string.push_str(&symbol.ident);
    }
}

pub fn dump(symbol: &Symbol) -> String {
    let special_symbols = SpecialSymbols {
        infix: hashmap! {
            "+" => Precedence::PSum,
            "-" => Precedence::PSum,
            "*" => Precedence::PProduct,
            "/" => Precedence::PProduct,
            "^" => Precedence::PPower,
        },
        postfix: vec!["!"].into_iter().collect(),
    };
    let mut string = String::new();
    dump_impl(&special_symbols, symbol, &mut string);
    string
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", dump(self))
    }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::context::*;
    use std::collections::HashMap;

    fn create_context(function_names: Vec<&str>) -> Context {
        let mut functions: HashMap<String, Declaration> = HashMap::new();
        for function_name in function_names.iter() {
            functions.insert(
                String::from(*function_name),
                Declaration {
                    is_fixed: false,
                    is_function: true,
                },
            );
        }
        Context { functions }
    }

    #[test]
    fn function_simple() {
        let context = create_context(vec!["f"]);
        let term = Symbol::parse(&context, "f(a)");
        assert_eq!(dump(&term), String::from("f(a)"));
    }

    #[test]
    fn infix_simple() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a+b");
        assert_eq!(dump(&term), String::from("a+b"));
    }

    #[test]
    fn infix_precedence() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a+b*c");
        assert_eq!(dump(&term), String::from("a+b*c"));
    }

    #[test]
    fn infix_parenthesis() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "(a+b)*c");
        assert_eq!(dump(&term), String::from("(a+b)*c"));
    }

    #[test]
    fn postfix_simple() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a!");
        assert_eq!(dump(&term), String::from("a!"));
    }

    #[test]
    fn postfix_with_infix() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "(a+b)!");
        assert_eq!(dump(&term), String::from("(a+b)!"));
    }
}
