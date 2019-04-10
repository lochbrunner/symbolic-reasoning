//! Composing tokens to an AST
//! Following https://de.wikipedia.org/wiki/Shunting-yard-Algorithmus

use crate::context::*;
use crate::parser::token::*;
use crate::symbol::Symbol;
use std::slice::Iter;

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PSeperator,
    PCall,
    POpening,
    // PClosing,
    PEquals,
    PLessGreater,
    PSum,
    PProduct,
    PPower,
    PFaculty,
    // PHighest,
}
#[derive(Debug, PartialEq)]
enum ParansDirection {
    Closing,
    Opening,
}
#[derive(Debug, PartialEq)]
enum ParansType {
    Round,
}
#[derive(Debug, PartialEq)]
struct Parans {
    pub direction: ParansDirection,
    pub r#type: ParansType,
}

#[derive(Debug, PartialEq)]
enum Classification {
    Infix(Operation),
    Prefix(Operation),
    Postfix(Operation),
    Parans(Parans),
    Separator,
    Ident(String),
    Literal(i64),
    EOF,
}

fn create_infix(ident: &str, precedence: Precedence) -> Option<Result<Classification, String>> {
    Some(Ok(Classification::Infix(Operation {
        ident: String::from(ident),
        precedence,
        r#type: OperationType::Infix,
    })))
}

fn create_prefix(ident: &str, precedence: Precedence) -> Option<Result<Classification, String>> {
    Some(Ok(Classification::Prefix(Operation {
        ident: String::from(ident),
        precedence,
        r#type: OperationType::Prefix,
    })))
}

fn create_function(ident: &str, precedence: Precedence) -> Option<Result<Classification, String>> {
    Some(Ok(Classification::Prefix(Operation {
        ident: String::from(ident),
        precedence,
        r#type: OperationType::Function,
    })))
}

fn create_postfix(ident: &str, precedence: Precedence) -> Option<Result<Classification, String>> {
    Some(Ok(Classification::Postfix(Operation {
        ident: String::from(ident),
        precedence,
        r#type: OperationType::Postfix,
    })))
}

mod token_type {
    pub const PREFIX: u32 = 1 << 0;
    pub const INFIX: u32 = 1 << 1;
    pub const POSTFIX: u32 = 1 << 2;
    pub const IDENT: u32 = 1 << 3;
    pub const LITERAL: u32 = 1 << 4;
    pub const CLOSING_PARENS: u32 = 1 << 5;
    pub const OPENING_PARENS: u32 = 1 << 6;
    pub const SEPARATOR: u32 = 1 << 7;
    pub const EOF: u32 = 1 << 8;
}

fn token_info(token: &Token) -> (u32, &'static str, Precedence, Option<&String>, Option<&i64>) {
    use token_type::*;
    match token {
        Token::Number(value) => (LITERAL, "", Precedence::PLowest, None, Some(value)),
        Token::Ident(ident) => (IDENT, "", Precedence::PLowest, Some(ident), None),
        Token::Plus => (INFIX | PREFIX, "+", Precedence::PSum, None, None),
        Token::Minus => (INFIX | PREFIX, "-", Precedence::PSum, None, None),
        Token::Multiply => (INFIX, "*", Precedence::PProduct, None, None),
        Token::Divide => (INFIX, "/", Precedence::PProduct, None, None),
        Token::Power => (INFIX, "^", Precedence::PPower, None, None),
        Token::Equal => (INFIX, "==", Precedence::PEquals, None, None),
        Token::NotEqual => (INFIX, "!=", Precedence::PEquals, None, None),
        Token::GreaterThan => (INFIX, ">", Precedence::PLessGreater, None, None),
        Token::LessThan => (INFIX, "<", Precedence::PLessGreater, None, None),
        Token::GreaterThanEqual => (INFIX, ">=", Precedence::PLessGreater, None, None),
        Token::LessThanEqual => (INFIX, "<=", Precedence::PLessGreater, None, None),
        Token::ParenL => (OPENING_PARENS, "(", Precedence::POpening, None, None),
        Token::ParenR => (CLOSING_PARENS, ")", Precedence::POpening, None, None),
        Token::Comma => (SEPARATOR, ",", Precedence::PLowest, None, None),
        Token::Faculty => (POSTFIX, "!", Precedence::PFaculty, None, None),
        Token::EOF => (EOF, "", Precedence::PLowest, None, None),
    }
}

/// Purpose
/// State-full iteration over tokens
/// * Distinguishes if operator is prefix of infix
/// * Fill default operator for instance ab => a*b
struct Classifier<'a> {
    tokens: Iter<'a, Token>,
    expect_operator: bool,
    next: Option<&'a Token>,
    context: &'a Context,
}

struct Tokens<'a>(&'a Vec<Token>);

impl<'a> Tokens<'a> {
    pub fn iter(&'a self, context: &'a Context) -> Classifier<'a> {
        Classifier {
            tokens: self.0.iter(),
            expect_operator: false,
            next: None,
            context,
        }
    }
}

impl<'a> Iterator for Classifier<'a> {
    type Item = Result<Classification, String>;

    fn next(&mut self) -> Option<Result<Classification, String>> {
        let next = if let Some(token) = self.next {
            Some(token)
        } else {
            self.tokens.next()
        };
        self.next = None;

        match next {
            Some(token) => {
                let token_info = token_info(token);
                if self.expect_operator {
                    if token_info.0 & token_type::INFIX != 0 {
                        self.expect_operator = false;
                        create_infix(token_info.1, token_info.2)
                    } else if token_info.0 & token_type::POSTFIX != 0 {
                        create_postfix(token_info.1, token_info.2)
                    } else if token_info.0 & token_type::EOF != 0 {
                        Some(Ok(Classification::EOF))
                    } else if token_info.0 & token_type::CLOSING_PARENS != 0 {
                        Some(Ok(Classification::Parans(Parans {
                            direction: ParansDirection::Closing,
                            r#type: ParansType::Round,
                        })))
                    } else if token_info.0 & token_type::SEPARATOR != 0 {
                        self.expect_operator = false;
                        Some(Ok(Classification::Separator))
                    } else if token_info.0 & (token_type::IDENT | token_type::OPENING_PARENS) != 0 {
                        self.expect_operator = false;
                        self.next = Some(token);
                        create_infix("*", Precedence::PProduct)
                    } else {
                        Some(Err(format!("Expected operator, found {:?}", token)))
                    }
                } else {
                    if token_info.0 & token_type::IDENT != 0 {
                        // Is this a function?
                        let ident = token_info.3.expect("Ident in tuple").clone();
                        if self.context.is_function(&ident) {
                            create_function(&ident[..], Precedence::PCall)
                        } else {
                            self.expect_operator = true;
                            Some(Ok(Classification::Ident(ident)))
                        }
                    } else if token_info.0 & token_type::LITERAL != 0 {
                        self.expect_operator = true;
                        let value = token_info.4.expect("Value in tuple").clone();
                        Some(Ok(Classification::Literal(value)))
                    } else if token_info.0 & token_type::PREFIX != 0 {
                        create_prefix(token_info.1, token_info.2)
                    } else if token_info.0 & token_type::EOF != 0 {
                        Some(Ok(Classification::EOF))
                    } else if token_info.0 & token_type::OPENING_PARENS != 0 {
                        Some(Ok(Classification::Parans(Parans {
                            direction: ParansDirection::Opening,
                            r#type: ParansType::Round,
                        })))
                    } else if token_info.0 & token_type::CLOSING_PARENS != 0 {
                        self.expect_operator = true;
                        Some(Ok(Classification::Parans(Parans {
                            direction: ParansDirection::Closing,
                            r#type: ParansType::Round,
                        })))
                    } else {
                        Some(Err(format!("Expected literal or ident, found {:?}", token)))
                    }
                }
            }
            None => None,
        }
    }
}

#[derive(Debug, PartialEq)]
enum OperationType {
    Infix,
    Prefix,
    Postfix,
    Function,
    Dummy,
}

#[derive(Debug, PartialEq)]
struct Operation {
    precedence: Precedence,
    ident: String,
    r#type: OperationType,
}

#[derive(Debug)]
enum IdentOrSymbol {
    Ident(String),
    Symbol(Symbol),
}

fn pop_as_symbol(sym_stack: &mut Vec<IdentOrSymbol>) -> Symbol {
    match sym_stack.pop().expect("Getting symbol") {
        IdentOrSymbol::Ident(ident) => Symbol::new_variable_from_string(ident),
        IdentOrSymbol::Symbol(symbol) => symbol,
    }
}

struct ParseStack {
    pub symbol: Vec<IdentOrSymbol>,
    pub infix: Vec<Operation>,
}

fn astify(context: &Context, stack: &mut ParseStack, till: Precedence) {
    while !stack.infix.is_empty() && stack.infix.last().expect("infix").precedence > till {
        match stack.infix.pop().unwrap() {
            Operation { ident, r#type, .. } => {
                let childs = match r#type {
                    OperationType::Infix => {
                        let b = pop_as_symbol(&mut stack.symbol);
                        let a = pop_as_symbol(&mut stack.symbol);
                        vec![a, b] // Order has to be reverted
                    }
                    OperationType::Prefix => vec![pop_as_symbol(&mut stack.symbol)],
                    _ => panic!("Invalid argument count {:?}", r#type),
                };
                stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
                    depth: Symbol::calc_depth(&childs),
                    fixed: context.is_fixed(&ident),
                    ident,
                    childs,
                    value: None,
                }));
            }
        }
    }
}

fn apply_function(context: &Context, stack: &mut ParseStack) {
    // Create function
    // TODO: Merge this with astify later
    let mut childs = vec![];
    childs.push(pop_as_symbol(&mut stack.symbol));
    while stack.infix.pop().expect("Something").precedence == Precedence::PSeperator {
        childs.push(pop_as_symbol(&mut stack.symbol));
    }
    childs.reverse();

    // Was this a function call?
    if !stack.infix.is_empty()
        && stack.infix.last().expect("Some infix").r#type == OperationType::Function
    {
        let func = stack.infix.pop().expect("Some infix");
        stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
            fixed: context.is_fixed(&func.ident),
            depth: Symbol::calc_depth(&childs),
            ident: func.ident,
            childs,
            value: None,
        }));
    } else {
        assert_eq!(
            childs.len(),
            1,
            "Expecting group containing one item. Vectors not supported yet!"
        );
        stack
            .symbol
            .push(IdentOrSymbol::Symbol(childs.pop().unwrap()));
    }
}

fn apply_postfix(context: &Context, stack: &mut ParseStack, ident: String) {
    let childs = vec![pop_as_symbol(&mut stack.symbol)];
    stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
        fixed: context.is_fixed(&ident),
        depth: Symbol::calc_depth(&childs),
        ident,
        childs,
        value: None,
    }));
}

pub fn parse(context: &Context, tokens: &Vec<Token>) -> Symbol {
    let mut stack = ParseStack {
        infix: Vec::new(),
        symbol: Vec::new(),
    };

    let tokens = Tokens(tokens);

    for token in tokens.iter(context) {
        match token {
            Ok(token) => match token {
                Classification::Infix(operation) => {
                    if let Some(last) = stack.infix.last() {
                        if last.precedence > operation.precedence {
                            astify(context, &mut stack, operation.precedence.clone());
                        }
                    }
                    stack.infix.push(operation);
                }
                Classification::Prefix(operation) => stack.infix.push(operation),
                Classification::Postfix(operation) => {
                    apply_postfix(context, &mut stack, operation.ident)
                }
                Classification::Ident(ident) => stack.symbol.push(IdentOrSymbol::Ident(ident)),
                Classification::Literal(value) => stack
                    .symbol
                    .push(IdentOrSymbol::Symbol(Symbol::new_number(value))),
                Classification::EOF => break,
                Classification::Parans(parans) => match parans.direction {
                    ParansDirection::Closing => {
                        astify(context, &mut stack, Precedence::POpening);
                        apply_function(context, &mut stack);
                    }
                    ParansDirection::Opening => stack.infix.push(Operation {
                        precedence: Precedence::POpening,
                        ident: String::from("("),
                        r#type: OperationType::Dummy,
                    }),
                },
                Classification::Separator => {
                    stack.infix.push(Operation {
                        precedence: Precedence::PSeperator,
                        ident: String::from(","),
                        r#type: OperationType::Dummy,
                    });
                }
            },
            Err(err) => panic!(err),
        };
    }

    astify(context, &mut stack, Precedence::PLowest);
    assert!(stack.infix.is_empty());
    assert_eq!(stack.symbol.len(), 1);
    return pop_as_symbol(&mut stack.symbol);
}

#[cfg(test)]
mod specs {
    use super::*;
    use std::collections::HashMap;
    use test::Bencher;

    fn create_function(ident: &str) -> Classification {
        Classification::Prefix(Operation {
            precedence: Precedence::PCall,
            ident: String::from(ident),
            r#type: OperationType::Function,
        })
    }

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
    fn classifier_single_ident_no_args() {
        let context = Context {
            functions: HashMap::new(),
        };
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let tokens = Tokens(&tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();
        assert_eq!(
            actual,
            vec![
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::EOF)
            ]
        );
    }

    #[test]
    fn single_ident_no_args() {
        let context = Context {
            functions: HashMap::new(),
        };
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let actual = parse(&context, &tokens);
        assert_eq!(actual, Symbol::new_variable("a"));
    }

    #[test]
    fn classifier_function_with_single_arg() {
        let context = create_context(vec!["f"]);

        let raw_tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::EOF,
        ];
        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(create_function("f")),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Opening,
                    r#type: ParansType::Round
                })),
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Closing,
                    r#type: ParansType::Round
                })),
                Ok(Classification::EOF)
            ]
        );
    }

    #[test]
    fn function_with_single_arg() {
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator("f", vec![Symbol::new_variable("a")])
        );
    }

    #[test]
    fn classifier_function_with_multiple_args() {
        let context = create_context(vec!["f"]);
        let raw_tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Comma,
            Token::Ident(String::from("b")),
            Token::Comma,
            Token::Ident(String::from("c")),
            Token::ParenR,
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(create_function("f")),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Opening,
                    r#type: ParansType::Round
                })),
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Separator),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::Separator),
                Ok(Classification::Ident(String::from("c"))),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Closing,
                    r#type: ParansType::Round
                })),
                Ok(Classification::EOF)
            ]
        );
    }

    #[test]
    fn function_with_multiple_args() {
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Comma,
            Token::Ident(String::from("b")),
            Token::Comma,
            Token::Ident(String::from("c")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);

        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                vec![
                    Symbol::new_variable("a"),
                    Symbol::new_variable("b"),
                    Symbol::new_variable("c")
                ]
            )
        );
    }

    #[test]
    fn function_nested_simple() {
        let context = create_context(vec!["f", "g"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("g")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                vec![Symbol::new_operator("g", vec![Symbol::new_variable("a")])]
            )
        );
    }

    #[test]
    fn function_inner_operator() {
        // f(a + b)
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                vec![Symbol::new_operator(
                    "+",
                    vec![Symbol::new_variable("a"), Symbol::new_variable("b"),]
                )]
            )
        );
    }

    #[bench]
    fn function_nested_with_inner_operator(b: &mut Bencher) {
        // f(g(a) + h(b))
        let context = create_context(vec!["f", "g", "h"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("g")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::Plus,
            Token::Ident(String::from("h")),
            Token::ParenL,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                vec![Symbol::new_operator(
                    "+",
                    vec![
                        Symbol::new_operator("g", vec![Symbol::new_variable("a")]),
                        Symbol::new_operator("h", vec![Symbol::new_variable("b"),])
                    ]
                )]
            )
        );

        b.iter(|| {
            parse(&context, &tokens);
        })
    }

    #[test]
    fn classify_bin_operator_simple() {
        let context = create_context(vec![]);
        let raw_tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Infix(Operation {
                    ident: String::from("+"),
                    precedence: Precedence::PSum,
                    r#type: OperationType::Infix,
                })),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::EOF)
            ]
        );
    }

    #[test]
    fn bin_operator_simple() {
        let context = create_context(vec![]);
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "+",
                vec![Symbol::new_variable("a"), Symbol::new_variable("b")]
            )
        );
    }

    #[test]
    fn bin_operator_order_of_operations() {
        // a+b*c-d
        let context = create_context(vec![]);
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::Minus,
            Token::Ident(String::from("d")),
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);

        assert_eq!(
            actual,
            Symbol::new_operator(
                "+",
                vec![
                    Symbol::new_variable("a"),
                    Symbol::new_operator(
                        "-",
                        vec![
                            Symbol::new_operator(
                                "*",
                                vec![Symbol::new_variable("b"), Symbol::new_variable("c")]
                            ),
                            Symbol::new_variable("d")
                        ]
                    ),
                ]
            )
        );
    }

    #[test]
    fn operators_with_parens_front() {
        // (a+b)*c
        let context = create_context(vec![]);
        let tokens = vec![
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "*",
            vec![
                Symbol::new_operator(
                    "+",
                    vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
                ),
                Symbol::new_variable("c"),
            ],
        );
        assert_eq!(actual.to_string(), expected.to_string());
        assert_eq!(actual, expected);
    }

    #[test]
    fn classifier_infix_operator_and_function() {
        // a + f(b)
        let context = create_context(vec!["f"]);
        let raw_tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Infix(Operation {
                    ident: String::from("+"),
                    precedence: Precedence::PSum,
                    r#type: OperationType::Infix
                })),
                Ok(create_function("f")),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Opening,
                    r#type: ParansType::Round
                })),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::Parans(Parans {
                    direction: ParansDirection::Closing,
                    r#type: ParansType::Round
                })),
                Ok(Classification::EOF)
            ]
        );
    }

    #[test]
    fn infix_operator_and_function() {
        // a + f(b)
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "+",
            vec![
                Symbol::new_variable("a"),
                Symbol::new_operator("f", vec![Symbol::new_variable("b")]),
            ],
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn classifier_prefix_operator_simple() {
        // -a
        let context = create_context(vec![]);
        let raw_tokens = vec![Token::Minus, Token::Ident(String::from("a")), Token::EOF];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(Classification::Prefix(Operation {
                    ident: String::from("-"),
                    precedence: Precedence::PSum,
                    r#type: OperationType::Prefix,
                })),
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::EOF),
            ]
        );
    }

    #[test]
    fn prefix_operator_simple() {
        // -a
        let context = create_context(vec![]);
        let tokens = vec![Token::Minus, Token::Ident(String::from("a")), Token::EOF];

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator("-", vec![Symbol::new_variable("a")]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn classifier_implicit_bin_operator_simple() {
        // ab -> a*b
        let context = create_context(vec![]);
        let raw_tokens = vec![
            Token::Ident(String::from("a")),
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&context)
            .collect::<Vec<Result<Classification, String>>>();

        assert_eq!(
            actual,
            vec![
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Infix(Operation {
                    ident: String::from("*"),
                    precedence: Precedence::PProduct,
                    r#type: OperationType::Infix,
                })),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::EOF),
            ]
        );
    }

    #[test]
    fn implicit_bin_operator_simple() {
        // ab -> a*b
        let context = create_context(vec![]);
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "*",
            vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
        );
        assert_eq!(actual, expected);
    }

    #[bench]
    fn implicit_bin_operator_parans(b: &mut Bencher) {
        // ab -> (a+b)(c+d)*e(f+g)
        let context = create_context(vec![]);
        let tokens = vec![
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::ParenL,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::Ident(String::from("d")),
            Token::ParenR,
            Token::Multiply,
            Token::Ident(String::from("e")),
            Token::ParenL,
            Token::Ident(String::from("f")),
            Token::Plus,
            Token::Ident(String::from("g")),
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "*",
            vec![
                Symbol::new_operator(
                    "+",
                    vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
                ),
                Symbol::new_operator(
                    "*",
                    vec![
                        Symbol::new_operator(
                            "+",
                            vec![Symbol::new_variable("c"), Symbol::new_variable("d")],
                        ),
                        Symbol::new_operator(
                            "*",
                            vec![
                                Symbol::new_variable("e"),
                                Symbol::new_operator(
                                    "+",
                                    vec![Symbol::new_variable("f"), Symbol::new_variable("g")],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        );
        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens);
        })
    }

    #[bench]
    fn function_complex_inner(b: &mut Bencher) {
        // f((a+b)*c+d*(e+h))
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::Ident(String::from("d")),
            Token::Multiply,
            Token::ParenL,
            Token::Ident(String::from("e")),
            Token::Plus,
            Token::Ident(String::from("h")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);

        let expected = Symbol::new_operator(
            "f",
            vec![Symbol::new_operator(
                "+",
                vec![
                    Symbol::new_operator(
                        "*",
                        vec![
                            Symbol::new_operator(
                                "+",
                                vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
                            ),
                            Symbol::new_variable("c"),
                        ],
                    ),
                    Symbol::new_operator(
                        "*",
                        vec![
                            Symbol::new_variable("d"),
                            Symbol::new_operator(
                                "+",
                                vec![Symbol::new_variable("e"), Symbol::new_variable("h")],
                            ),
                        ],
                    ),
                ],
            )],
        );

        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens);
        })
    }

    #[test]
    fn double_parens() {
        // ((a+b))
        let context = create_context(vec![]);
        let tokens = vec![
            Token::ParenL,
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "+",
            vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn function_double_parens() {
        // f((a))
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator("f", vec![Symbol::new_variable("a")]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn prefix_operator() {
        // a*-b
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Multiply,
            Token::Minus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "*",
            vec![
                Symbol::new_variable("a"),
                Symbol::new_operator("-", vec![Symbol::new_variable("b")]),
            ],
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn postfix_operator_simple() {
        // a!
        let tokens = vec![Token::Ident(String::from("a")), Token::Faculty, Token::EOF];
        let context = create_context(vec![]);

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator("!", vec![Symbol::new_variable("a")]);

        assert_eq!(actual, expected);
    }

    #[bench]
    fn postfix_operator_complex(b: &mut Bencher) {
        // a+b!*c+(e*d)!
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::Faculty,
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::ParenL,
            Token::Ident(String::from("e")),
            Token::Multiply,
            Token::Ident(String::from("d")),
            Token::ParenR,
            Token::Faculty,
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "+",
            vec![
                Symbol::new_variable("a"),
                Symbol::new_operator(
                    "+",
                    vec![
                        Symbol::new_operator(
                            "*",
                            vec![
                                Symbol::new_operator("!", vec![Symbol::new_variable("b")]),
                                Symbol::new_variable("c"),
                            ],
                        ),
                        Symbol::new_operator(
                            "!",
                            vec![Symbol::new_operator(
                                "*",
                                vec![Symbol::new_variable("e"), Symbol::new_variable("d")],
                            )],
                        ),
                    ],
                ),
            ],
        );

        assert_eq!(actual, expected);
        b.iter(|| {
            parse(&context, &tokens);
        })
    }

    #[bench]
    fn precedence_pyramid(b: &mut Bencher) {
        // a*b^c+d^e*f
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Multiply,
            Token::Ident(String::from("b")),
            Token::Power,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::Ident(String::from("d")),
            Token::Power,
            Token::Ident(String::from("e")),
            Token::Multiply,
            Token::Ident(String::from("f")),
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens);
        let expected = Symbol::new_operator(
            "+",
            vec![
                Symbol::new_operator(
                    "*",
                    vec![
                        Symbol::new_variable("a"),
                        Symbol::new_operator(
                            "^",
                            vec![Symbol::new_variable("b"), Symbol::new_variable("c")],
                        ),
                    ],
                ),
                Symbol::new_operator(
                    "*",
                    vec![
                        Symbol::new_operator(
                            "^",
                            vec![Symbol::new_variable("d"), Symbol::new_variable("e")],
                        ),
                        Symbol::new_variable("f"),
                    ],
                ),
            ],
        );

        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens);
        })
    }

    #[test]
    fn numbers() {
        // 1+2*3
        let tokens = vec![
            Token::Number(1),
            Token::Plus,
            Token::Number(2),
            Token::Multiply,
            Token::Number(3),
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens);

        let expected = Symbol::new_operator(
            "+",
            vec![
                Symbol::new_number(1),
                Symbol::new_operator("*", vec![Symbol::new_number(2), Symbol::new_number(3)]),
            ],
        );

        assert_eq!(actual, expected);
    }
}
