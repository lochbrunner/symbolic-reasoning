use super::base::{dump_base, SpecialFormatRules, SpecialSymbols};
use crate::parser::Precedence;
use crate::Symbol;
use std::fmt;

pub fn dump_simple(symbol: &Symbol) -> String {
    let special_symbols = SpecialSymbols {
        infix: hashmap! {
            "+" => Precedence::PSum,
            "-" => Precedence::PSum,
            "*" => Precedence::PProduct,
            "/" => Precedence::PProduct,
            "^" => Precedence::PPower,
            "=" => Precedence::PEquals,
            "==" => Precedence::PEquals,
            "!=" => Precedence::PEquals,
        },
        postfix: hashset! {"!"},
        prefix: hashset! {"-"},
        format: SpecialFormatRules {
            symbols: hashmap! {},
            functions: hashmap! {},
        },
    };
    let mut string = String::new();
    dump_base(&special_symbols, symbol, &mut string);
    string
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", dump_simple(self))
    }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::context::*;
    use std::collections::HashMap;

    fn create_context(function_names: Vec<&str>) -> Context {
        let mut declarations: HashMap<String, Declaration> = HashMap::new();
        for function_name in function_names.iter() {
            declarations.insert(
                String::from(*function_name),
                Declaration {
                    is_fixed: false,
                    is_function: true,
                    only_root: false,
                },
            );
        }
        Context { declarations }
    }

    #[test]
    fn function_simple() {
        let context = create_context(vec!["f"]);
        let term = Symbol::parse(&context, "f(a)");
        assert_eq!(dump_simple(&term), String::from("f(a)"));
    }

    #[test]
    fn infix_simple() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a+b");
        assert_eq!(dump_simple(&term), String::from("a+b"));
    }

    #[test]
    fn infix_precedence() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a+b*c");
        assert_eq!(dump_simple(&term), String::from("a+b*c"));
    }

    #[test]
    fn infix_parenthesis() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "(a+b)*c");
        assert_eq!(dump_simple(&term), String::from("(a+b)*c"));
    }

    #[test]
    fn postfix_simple() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a!");
        assert_eq!(dump_simple(&term), String::from("a!"));
    }

    #[test]
    fn postfix_with_infix() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "(a+b)!");
        assert_eq!(dump_simple(&term), String::from("(a+b)!"));
    }
}
