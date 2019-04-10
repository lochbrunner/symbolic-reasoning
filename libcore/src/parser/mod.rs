mod lexer;
mod parser;
mod token;

use super::symbol::Symbol;
use crate::context::Context;

impl Symbol {
    pub fn parse_from_str(context: &Context, code: String) -> Symbol {
        let (_, tokens) = lexer::lex_tokens(code.as_bytes()).expect("tokens");
        parser::parse(context, &tokens)
    }

    pub fn parse(context: &Context, code: &str) -> Symbol {
        let (_, tokens) = lexer::lex_tokens(code.as_bytes()).expect("tokens");
        parser::parse(context, &tokens)
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
    fn operator_simple() {
        let context = create_context(vec![]);
        let actual = Symbol::parse(&context, "a+b*c");

        assert_eq!(
            actual,
            Symbol::new_operator(
                "+",
                vec![
                    Symbol::new_variable("a"),
                    Symbol::new_operator(
                        "*",
                        vec![Symbol::new_variable("b"), Symbol::new_variable("c"),]
                    )
                ]
            )
        );
    }
}
