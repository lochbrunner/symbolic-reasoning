use crate::parser::token::*;
use crate::symbol::Symbol;
use nom::*;
use std::str::FromStr;

mod lexer;
pub mod token;

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PSeperator,
    POpening,
    PClosing,
    PEquals,
    PLessGreater,
    PSum,
    PProduct,
    PHighest,
    // PIndex,
}

type Tokens = Vec<Token>;

fn classify(token: &Token) -> (Option<Operation>, Option<String>) {
    match token {
        Token::Minus => (
            Some(Operation::create_op("-", token.clone(), Precedence::PSum)),
            None,
        ),
        Token::Plus => (
            Some(Operation::create_op("+", token.clone(), Precedence::PSum)),
            None,
        ),
        Token::Multiply => (
            Some(Operation::create_op(
                "*",
                token.clone(),
                Precedence::PProduct,
            )),
            None,
        ),
        Token::Divide => (
            Some(Operation::create_op(
                "/",
                token.clone(),
                Precedence::PProduct,
            )),
            None,
        ),
        Token::Equal => (
            Some(Operation::create_op(
                "==",
                token.clone(),
                Precedence::PEquals,
            )),
            None,
        ),
        Token::NotEqual => (
            Some(Operation::create_op(
                "!=",
                token.clone(),
                Precedence::PEquals,
            )),
            None,
        ),
        Token::GreaterThan => (
            Some(Operation::create_op(
                ">",
                token.clone(),
                Precedence::PLessGreater,
            )),
            None,
        ),
        Token::LessThan => (
            Some(Operation::create_op(
                "<",
                token.clone(),
                Precedence::PLessGreater,
            )),
            None,
        ),
        Token::GreaterThanEqual => (
            Some(Operation::create_op(
                ">=",
                token.clone(),
                Precedence::PLessGreater,
            )),
            None,
        ),
        Token::LessThanEqual => (
            Some(Operation::create_op(
                "<=",
                token.clone(),
                Precedence::PLessGreater,
            )),
            None,
        ),
        Token::ParenL => (
            Some(Operation::create_op(
                "(",
                token.clone(),
                Precedence::POpening,
            )),
            None,
        ),
        Token::ParenR => (
            Some(Operation {
                ident: String::from(")"),
                precedence: Precedence::PClosing,
                separator: Some(Token::Comma),
                closer: Some(Token::ParenL),
                token: token.clone(),
            }),
            None,
        ),
        Token::Comma => (
            Some(Operation::create_op(
                ",",
                token.clone(),
                Precedence::PLowest,
            )),
            None,
        ),
        Token::Ident(ident) => (None, Some(ident.clone())),
        Token::EOF => (None, None),
        _ => panic!("No arm implemented for token {:?} !", token),
    }
}

#[derive(Debug)]
struct Operation {
    precedence: Precedence,
    token: Token,
    separator: Option<Token>,
    closer: Option<Token>,
    ident: String,
}

impl Operation {
    pub fn create_op(ident: &str, token: Token, precedence: Precedence) -> Operation {
        Operation {
            precedence,
            ident: String::from_str(ident).unwrap(),
            separator: None,
            closer: None,
            token,
        }
    }
}

#[derive(Debug)]
enum IdentOrSymbol {
    Ident(String),
    Symbol(Symbol),
}

fn pop_as_symbol(sym_stack: &mut Vec<IdentOrSymbol>) -> Symbol {
    return match sym_stack.pop().expect("Getting symbol") {
        IdentOrSymbol::Ident(ident) => Symbol::new_variable_by_string(ident),
        IdentOrSymbol::Symbol(symbol) => symbol,
    };
}

// fn print_op_stack(op_stack: &Vec<Operation>) {
//     print!("operations: ");
//     for op in op_stack.iter() {
//         print!("{} ", op.ident);
//     }
//     print!("\n");
// }

// fn print_sym_stack(stack: &Vec<IdentOrSymbol>) {
//     print!("symbols: ");
//     for item in stack.iter() {
//         match item {
//             IdentOrSymbol::Ident(ident) => print!("\"{}\" ", ident),
//             IdentOrSymbol::Symbol(symbol) => print!("{} ", symbol),
//         };
//     }
//     print!("\n");
// }

fn astify(op_stack: &mut Vec<Operation>, sym_stack: &mut Vec<IdentOrSymbol>) {
    while !op_stack.is_empty() && op_stack.last().unwrap().token != Token::ParenL {
        match op_stack.pop().unwrap() {
            Operation {
                precedence: _,
                separator: None,
                closer: None,
                ident,
                token: _,
            } => {
                // Only binary operators
                let b = pop_as_symbol(sym_stack);
                let a = pop_as_symbol(sym_stack);

                sym_stack.push(IdentOrSymbol::Symbol(Symbol::new_operator_by_string(
                    ident,
                    vec![a, b],
                )));
            }
            Operation {
                precedence: _,
                separator: Some(separator),
                closer: Some(closer),
                ident: _,
                token: _,
            } => {
                let mut childs: Vec<Symbol> = vec![pop_as_symbol(sym_stack)];
                'childs: loop {
                    let op = op_stack.pop().expect("Getting from operation stack");
                    if op.token == closer {
                        println!("Close with {:?}", op.token);
                        break 'childs;
                    } else if op.token == separator {
                        println!("Sep with {:?}", op.token);
                        childs.push(pop_as_symbol(sym_stack));
                    } else {
                        panic!("Unexpected operator {:?} found!", op);
                    }
                }
                // Function or standard brackets ?
                let ident = if op_stack.len() < sym_stack.len() {
                    match sym_stack.pop().expect("Getting symbol stack") {
                        IdentOrSymbol::Ident(ident) => ident,
                        IdentOrSymbol::Symbol(symbol) => {
                            panic!("Not expected symbol {:?} to be ident!", symbol)
                        }
                    }
                } else {
                    String::from("identity")
                };

                childs.reverse();
                sym_stack.push(IdentOrSymbol::Symbol(Symbol::new_operator_by_string(
                    ident, childs,
                )));
                return;
            }
            _ => panic!("Not implemented yet"),
        }
    }
}

pub fn parse(tokens: &Tokens) -> Symbol {
    let mut op_stack: Vec<Operation> = Vec::new();
    let mut sym_stack: Vec<IdentOrSymbol> = Vec::new();
    for token in tokens.iter() {
        match classify(token) {
            (_, Some(ident)) => sym_stack.push(IdentOrSymbol::Ident(ident)),
            (None, None) => {
                astify(&mut op_stack, &mut sym_stack);
                assert!(op_stack.is_empty());
                assert_eq!(sym_stack.len(), 1);
                return pop_as_symbol(&mut sym_stack);
            }
            (Some(op), None) => {
                if let Some(last) = op_stack.last() {
                    if last.precedence > op.precedence {
                        // astify
                        if op.token != Token::ParenL {
                            astify(&mut op_stack, &mut sym_stack);
                        }
                        op_stack.push(op);
                    } else if op.token == Token::ParenR {
                        op_stack.push(op);
                        astify(&mut op_stack, &mut sym_stack);
                    } else {
                        op_stack.push(op);
                    }
                } else {
                    op_stack.push(op);
                }
            }
        }
    }

    Symbol::new_variable("Hi")
}

#[cfg(test)]
mod specs {
    use super::*;

    #[test]
    fn single_ident_no_args() {
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let actual = parse(&tokens);
        assert_eq!(actual, Symbol::new_variable("a"));
    }

    #[test]
    fn function_with_single_arg() {
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::ParenL,
            Token::Ident(String::from("a")),
            Token::ParenR,
            Token::EOF,
        ];
        let actual = parse(&tokens);
        assert_eq!(
            actual,
            Symbol::new_operator("f", vec![Symbol::new_variable("a")])
        );
    }

    #[test]
    fn function_with_multiple_args() {
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
        let actual = parse(&tokens);
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

        let actual = parse(&tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                vec![Symbol::new_operator("g", vec![Symbol::new_variable("a")])]
            )
        );
    }

    #[test]
    fn function_nested_with_inner_operator() {
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

        let actual = parse(&tokens);
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
    fn bin_operator_simple() {
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::EOF,
        ];
        let actual = parse(&tokens);
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
        let actual = parse(&tokens);
        assert_eq!(
            actual,
            Symbol::new_operator(
                "-",
                vec![
                    Symbol::new_operator(
                        "+",
                        vec![
                            Symbol::new_variable("a"),
                            Symbol::new_operator(
                                "*",
                                vec![Symbol::new_variable("b"), Symbol::new_variable("c")]
                            )
                        ]
                    ),
                    Symbol::new_variable("d")
                ]
            )
        );
    }
}
