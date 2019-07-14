mod astifier;
mod lexer;
mod token;

use super::symbol::Symbol;
use crate::context::Context;

pub use astifier::Precedence;

impl Symbol {
    // pub fn parse_from_str(context: &Context, code: String) -> Symbol {
    //     let (_, tokens) = lexer::lex_tokens(code.as_bytes()).expect("tokens");
    //     astifier::parse(context, &tokens)
    // }

    pub fn parse(context: &Context, code: &str) -> Result<Symbol, String> {
        let (_, tokens) = lexer::lex_tokens(code.as_bytes()).expect("tokens");
        match astifier::parse(context, &tokens) {
            Ok(symbol) => Ok(symbol),
            Err(msg) => Err(format!("{}: {}", msg, code)),
        }
    }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::context::*;
    use std::collections::HashMap;

    fn new_variable(ident: &str) -> Symbol {
        Symbol::new_variable(ident, false)
    }

    fn new_op(ident: &str, childs: Vec<Symbol>) -> Symbol {
        Symbol::new_operator(ident, false, false, childs)
    }

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
    fn operator_simple() {
        let context = create_context(vec![]);
        let actual = Symbol::parse(&context, "a+b*c").unwrap();

        assert_eq!(
            actual,
            new_op(
                "+",
                vec![
                    new_variable("a"),
                    new_op("*", vec![new_variable("b"), new_variable("c"),])
                ]
            )
        );
    }

    #[test]
    fn equation_1() {
        // a - b == 0
        let context = create_context(vec![]);
        let actual = Symbol::parse(&context, "a - b = 0").unwrap();

        let expected = new_op(
            "=",
            vec![
                new_op("-", vec![new_variable("a"), new_variable("b")]),
                Symbol::new_number(0),
            ],
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn equation_2() {
        // a - b == 0
        let context = create_context(vec![]);
        let actual = Symbol::parse(&context, "a = b").unwrap();

        let expected = new_op("=", vec![new_variable("a"), new_variable("b")]);

        assert_eq!(actual, expected);
    }
    #[test]
    fn equation_3() {
        // x == -a
        let context = create_context(vec![]);
        let actual = Symbol::parse(&context, "x = -a").unwrap();

        let expected = new_op(
            "=",
            vec![new_variable("x"), new_op("-", vec![new_variable("a")])],
        );

        assert_eq!(actual, expected);
    }
}
