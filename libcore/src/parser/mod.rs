//! Composing tokens to an AST
//! Following https://de.wikipedia.org/wiki/Shunting-yard-Algorithmus

use crate::parser::token::*;
use crate::symbol::Symbol;
use std::collections::HashMap;
use std::slice::Iter;

mod lexer;
pub mod token;

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PSeperator,
    PCall,
    POpening,
    PClosing,
    PEquals,
    PLessGreater,
    PSum,
    PProduct,
    PPower,
    PFaculty,
    PHighest,
}
#[derive(Debug, PartialEq)]
enum ParansDirection {
    Closing,
    Opening,
}
#[derive(Debug, PartialEq)]
enum ParansType {
    Round,
    // Curly,
    // Square,
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
    // Literal(f64),
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
    pub const CLOSING_PARENS: u32 = 1 << 4;
    pub const OPENING_PARENS: u32 = 1 << 5;
    pub const SEPARATOR: u32 = 1 << 6;
    pub const EOF: u32 = 1 << 7;
}

fn token_info(token: &Token) -> (u32, &'static str, Precedence, Option<&String>) {
    use token_type::*;
    match token {
        Token::Ident(ident) => (IDENT, "", Precedence::PLowest, Some(ident)),
        Token::Plus => (INFIX | PREFIX, "+", Precedence::PSum, None),
        Token::Minus => (INFIX | PREFIX, "-", Precedence::PSum, None),
        Token::Multiply => (INFIX, "*", Precedence::PProduct, None),
        Token::Divide => (INFIX, "/", Precedence::PProduct, None),
        Token::Power => (INFIX, "^", Precedence::PPower, None),
        Token::Equal => (INFIX, "==", Precedence::PEquals, None),
        Token::NotEqual => (INFIX, "!=", Precedence::PEquals, None),
        Token::GreaterThan => (INFIX, ">", Precedence::PLessGreater, None),
        Token::LessThan => (INFIX, "<", Precedence::PLessGreater, None),
        Token::GreaterThanEqual => (INFIX, ">=", Precedence::PLessGreater, None),
        Token::LessThanEqual => (INFIX, "<=", Precedence::PLessGreater, None),
        Token::ParenL => (OPENING_PARENS, "(", Precedence::POpening, None),
        Token::ParenR => (CLOSING_PARENS, ")", Precedence::POpening, None),
        Token::Comma => (SEPARATOR, ",", Precedence::PLowest, None),
        Token::Faculty => (POSTFIX, "!", Precedence::PFaculty, None),
        Token::EOF => (EOF, "", Precedence::PLowest, None),
        _ => panic!("No arm implemented for token {:?} !", token),
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
    functions: &'a HashMap<String, ()>,
}

struct Tokens<'a>(&'a Vec<Token>);

impl<'a> Tokens<'a> {
    pub fn iter(&'a self, functions: &'a HashMap<String, ()>) -> Classifier<'a> {
        Classifier {
            tokens: self.0.iter(),
            expect_operator: false,
            next: None,
            functions,
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
                    } else if token_info.0 & token_type::IDENT != 0 {
                        self.expect_operator = false;
                        self.next = if let None = self.next {
                            Some(token)
                        } else {
                            None
                        };
                        create_infix("*", Precedence::PProduct)
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
                    } else if token_info.0 & token_type::OPENING_PARENS != 0 {
                        self.expect_operator = false;
                        self.next = if let None = self.next {
                            Some(token)
                        } else {
                            None
                        };
                        create_infix("*", Precedence::PProduct)
                    } else {
                        Some(Err(format!("Expected operator, found {:?}", token)))
                    }
                } else {
                    if token_info.0 & token_type::IDENT != 0 {
                        // Is this a function?
                        let ident = token_info.3.expect("Ident in tuple").clone();
                        self.next = None;
                        match self.functions.get_key_value(&ident) {
                            None => {
                                self.expect_operator = true;
                                Some(Ok(Classification::Ident(ident)))
                            }
                            // Treat functions as prefix operator
                            Some((key, _)) => {
                                self.expect_operator = false;
                                create_function(&key[..], Precedence::PCall)
                            }
                        }
                    } else if token_info.0 & token_type::PREFIX != 0 {
                        create_prefix(token_info.1, token_info.2)
                    } else if token_info.0 & token_type::EOF != 0 {
                        Some(Ok(Classification::EOF))
                    } else if token_info.0 & token_type::OPENING_PARENS != 0 {
                        self.expect_operator = false;
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
        IdentOrSymbol::Ident(ident) => Symbol::new_variable_by_string(ident),
        IdentOrSymbol::Symbol(symbol) => symbol,
    }
}

struct ParseStack {
    pub symbol: Vec<IdentOrSymbol>,
    pub infix: Vec<Operation>,
}

fn astify(stack: &mut ParseStack, till: Precedence) {
    // No infix?
    if stack.infix.is_empty() {
        let symbol = pop_as_symbol(&mut stack.symbol);
        stack.symbol.push(IdentOrSymbol::Symbol(symbol));
        return;
    }
    'infix: while !stack.infix.is_empty() && stack.infix.last().expect("infix").precedence > till {
        match stack.infix.pop().unwrap() {
            Operation { ident, r#type, .. } => {
                // Only binary operators
                let childs = match r#type {
                    OperationType::Infix => {
                        let b = pop_as_symbol(&mut stack.symbol);
                        let a = pop_as_symbol(&mut stack.symbol);
                        vec![a, b]
                    }
                    OperationType::Prefix => vec![pop_as_symbol(&mut stack.symbol)],
                    _ => panic!("Invalid argument count {:?}", r#type),
                };
                stack
                    .symbol
                    .push(IdentOrSymbol::Symbol(Symbol::new_operator_by_string(
                        ident, childs,
                    )));
            }

            _ => panic!("Not implemented yet"),
        }
    }
}

fn process_call(stack: &mut ParseStack) {
    // Create function
    // If no infix available, do nothing.
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
        let symbol = Symbol::new_operator_by_string(func.ident, childs);
        stack.symbol.push(IdentOrSymbol::Symbol(symbol));
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

fn apply_postfix(stack: &mut ParseStack, ident: String) {
    let childs = vec![pop_as_symbol(&mut stack.symbol)];
    let symbol = Symbol::new_operator_by_string(ident, childs);
    stack.symbol.push(IdentOrSymbol::Symbol(symbol));
}

fn print_stack(stack: &ParseStack) {
    println!("-----------");
    println!("Symbols:");
    for sym in stack.symbol.iter() {
        match sym {
            IdentOrSymbol::Ident(ident) => println!("Ident: {}", ident),
            IdentOrSymbol::Symbol(symbol) => println!("Symbol: {}", symbol),
            _ => println!("Unknown symbol {:?} found!", sym),
        }
    }

    println!("Operators:");
    for op in stack.infix.iter() {
        println!("operation: {} {:?}", op.ident, op.precedence);
    }
}

pub fn parse(functions: &HashMap<String, ()>, tokens: &Vec<Token>) -> Symbol {
    let mut stack = ParseStack {
        infix: Vec::new(),
        symbol: Vec::new(),
    };

    let tokens = Tokens(tokens);

    for token in tokens.iter(functions) {
        match token {
            Ok(token) => match token {
                Classification::Infix(operation) => {
                    if let Some(last) = stack.infix.last() {
                        if last.precedence > operation.precedence {
                            astify(&mut stack, operation.precedence.clone());
                        }
                    }
                    stack.infix.push(operation);
                }
                Classification::Prefix(operation) => stack.infix.push(operation),
                Classification::Postfix(operation) => apply_postfix(&mut stack, operation.ident),
                Classification::Ident(ident) => stack.symbol.push(IdentOrSymbol::Ident(ident)),
                Classification::EOF => break,
                Classification::Parans(parans) => match parans.direction {
                    ParansDirection::Closing => {
                        astify(&mut stack, Precedence::POpening);
                        process_call(&mut stack);
                    }
                    ParansDirection::Opening => stack.infix.push(Operation {
                        precedence: Precedence::POpening,
                        ident: String::from("("),
                        r#type: OperationType::Dummy,
                    }),
                },
                Classification::Separator => {
                    astify(&mut stack, Precedence::POpening);
                    stack.infix.push(Operation {
                        precedence: Precedence::PSeperator,
                        ident: String::from(","),
                        r#type: OperationType::Dummy,
                    });
                }
                _ => panic!("Arm {:?} is not implemented yet", token),
            },
            Err(err) => panic!(err),
        };
    }

    astify(&mut stack, Precedence::PLowest);
    assert!(stack.infix.is_empty());
    assert_eq!(stack.symbol.len(), 1);
    return pop_as_symbol(&mut stack.symbol);
}

#[cfg(test)]
mod specs {
    use super::*;
    use maplit::*;

    fn create_function(ident: &str) -> Classification {
        Classification::Prefix(Operation {
            precedence: Precedence::PCall,
            ident: String::from(ident),
            r#type: OperationType::Function,
        })
    }

    #[test]
    fn classifier_single_ident_no_args() {
        let functions = HashMap::new();
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let tokens = Tokens(&tokens);

        let actual = tokens
            .iter(&functions)
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
        let functions = HashMap::new();
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let actual = parse(&functions, &tokens);
        assert_eq!(actual, Symbol::new_variable("a"));
    }

    #[test]
    fn classifier_function_with_single_arg() {
        let functions = hashmap! {String::from("f")=> ()};
        let raw_tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::EOF,
        ];
        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&functions)
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
        let functions = hashmap! {String::from("f")=> ()};
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&functions, &tokens);
        assert_eq!(
            actual,
            Symbol::new_operator("f", vec![Symbol::new_variable("a")])
        );
    }

    #[test]
    fn classifier_function_with_multiple_args() {
        let functions = hashmap! {String::from("f")=> ()};
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
            .iter(&functions)
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
        let functions = hashmap! {String::from("f")=> ()};
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
        let actual = parse(&functions, &tokens);

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
        let functions = hashmap! {
            String::from("f")=> (),
            String::from("g")=> ()
        };
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

        let actual = parse(&functions, &tokens);
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
        let functions = hashmap! {
            String::from("f")=> (),
        };
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::EOF,
        ];

        let actual = parse(&functions, &tokens);
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

    #[test]
    fn function_nested_with_inner_operator() {
        // f(g(a) + h(b))
        let functions = hashmap! {
            String::from("f")=> (),
            String::from("g")=> (),
            String::from("h")=> (),
        };
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

        let actual = parse(&functions, &tokens);
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
    }

    #[test]
    fn classify_bin_operator_simple() {
        let functions = HashMap::new();
        let raw_tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&functions)
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
        let functions = HashMap::new();
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];
        let actual = parse(&functions, &tokens);
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
        let functions = HashMap::new();
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
        let actual = parse(&functions, &tokens);

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
        let functions = HashMap::new();
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
        let actual = parse(&functions, &tokens);
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
        let functions = hashmap! {
            String::from("f")=> ()
        };
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
            .iter(&functions)
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
        let functions = hashmap! {
            String::from("f")=> ()
        };
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("b")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&functions, &tokens);
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
        let functions = HashMap::new();
        let raw_tokens = vec![Token::Minus, Token::Ident(String::from("a")), Token::EOF];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&functions)
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
        let functions = HashMap::new();
        let tokens = vec![Token::Minus, Token::Ident(String::from("a")), Token::EOF];

        let actual = parse(&functions, &tokens);
        let expected = Symbol::new_operator("-", vec![Symbol::new_variable("a")]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn classifier_implicit_bin_operator_simple() {
        // ab -> a*b
        let functions = HashMap::new();
        let raw_tokens = vec![
            Token::Ident(String::from("a")),
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let tokens = Tokens(&raw_tokens);

        let actual = tokens
            .iter(&functions)
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
        let functions = HashMap::new();
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Ident(String::from("b")),
            Token::EOF,
        ];

        let actual = parse(&functions, &tokens);
        let expected = Symbol::new_operator(
            "*",
            vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn implicit_bin_operator_parans() {
        // ab -> (a+b)(c+d)*e(f+g)
        let functions = HashMap::new();
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

        let actual = parse(&functions, &tokens);
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
    }

    #[test]
    fn function_complex_inner() {
        // f((a+b)*c+d*(e+h))
        let functions = hashmap! {
            String::from("f")=> ()
        };
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

        let actual = parse(&functions, &tokens);

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
    }

    #[test]
    fn double_parens() {
        // ((a+b))
        let functions = HashMap::new();
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

        let actual = parse(&functions, &tokens);
        let expected = Symbol::new_operator(
            "+",
            vec![Symbol::new_variable("a"), Symbol::new_variable("b")],
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn function_double_parens() {
        // f((a))
        let functions = hashmap! {
            String::from("f")=> ()
        };
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&functions, &tokens);
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

        let functions = HashMap::new();

        let actual = parse(&functions, &tokens);
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
        let functions = HashMap::new();

        let actual = parse(&functions, &tokens);
        let expected = Symbol::new_operator("!", vec![Symbol::new_variable("a")]);

        assert_eq!(actual, expected);
    }

    #[test]
    fn postfix_operator_complex() {
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

        let functions = HashMap::new();

        let actual = parse(&functions, &tokens);
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
    }

    #[test]
    fn precedence_pyramid() {
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

        let functions = HashMap::new();

        let actual = parse(&functions, &tokens);
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
    }
}
