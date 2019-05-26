use super::base::*;
use crate::parser::Precedence;
use crate::Symbol;
use std::fmt;

pub fn dump_simple(symbol: &Symbol) -> String {
    let context = FormatContext {
        operators: Operators {
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
            non_associative: hashset! {"-","/"},
        },
        formats: SpecialFormatRules {
            symbols: hashmap! {},
            functions: hashmap! {},
        },
        decoration: None,
    };
    let mut string = String::new();
    dump_base(&context, symbol, FormatingLocation::new(), &mut string);
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

    fn create_context(function_names: &[&str]) -> Context {
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

    fn test(code: &str) {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, code);
        assert_eq!(dump_simple(&term), String::from(code));
    }

    fn test_with_function(function_names: &[&str], code: &str) {
        let context = create_context(function_names);
        let term = Symbol::parse(&context, code);
        assert_eq!(dump_simple(&term), String::from(code));
    }

    #[test]
    fn function_simple() {
        test_with_function(&["f"], "f(a)");
    }

    #[test]
    fn infix_simple() {
        test("a+b");
    }

    #[test]
    fn infix_precedence() {
        test("a+b*c");
    }

    #[test]
    fn infix_parenthesis() {
        test("(a+b)*c");
    }

    #[test]
    fn postfix_simple() {
        test("a!");
    }

    #[test]
    fn postfix_with_infix() {
        test("(a+b)!");
    }

    #[test]
    fn bug_15_substraction() {
        test("a-(b+c)");
    }

    #[test]
    fn bug_15_substraction_false_positive() {
        test("a-b");
    }

    #[test]
    fn bug_16() {
        test("(a+b)^c");
    }
}
