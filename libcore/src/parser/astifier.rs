//! Composing tokens to an AST
//! Following https://de.wikipedia.org/wiki/Shunting-yard-Algorithmus

use crate::context::*;
use crate::parser::token::*;
use crate::symbol::Symbol;
use std::slice::Iter;

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PSeparator,
    PCall,
    POpening,
    // PClosing,
    PEquals,
    PLessGreater,
    PSum,
    PProduct,
    PPower,
    PFaculty,
    PHighest,
}
#[derive(Debug, PartialEq)]
enum BracketDirection {
    Closing,
    Opening,
}
#[derive(Debug, PartialEq)]
enum BracketType {
    Round,
}
#[derive(Debug, PartialEq)]
struct Bracket {
    pub direction: BracketDirection,
    pub r#type: BracketType,
}

#[derive(Debug, PartialEq)]
enum Classification {
    Infix(Operation),
    Prefix(Operation),
    Postfix(Operation),
    Bracket(Bracket),
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
    pub const PREFIX: u32 = 1;
    pub const INFIX: u32 = 1 << 1;
    pub const POSTFIX: u32 = 1 << 2;
    pub const IDENT: u32 = 1 << 3;
    pub const LITERAL: u32 = 1 << 4;
    pub const CLOSING_BRACKET: u32 = 1 << 5;
    pub const OPENING_BRACKET: u32 = 1 << 6;
    pub const SEPARATOR: u32 = 1 << 7;
    pub const EOF: u32 = 1 << 8;
}

enum Payload<'a> {
    Ident(&'a String),
    Number(&'a i64),
}

fn token_info<'a>(token: &'a Token) -> (u32, &'static str, Precedence, Option<Payload<'a>>) {
    use token_type::*;
    match token {
        Token::Number(value) => (
            LITERAL,
            "",
            Precedence::PLowest,
            Some(Payload::Number(value)),
        ),
        Token::Ident(ident) => (IDENT, "", Precedence::PLowest, Some(Payload::Ident(ident))),
        Token::Plus => (INFIX | PREFIX, "+", Precedence::PSum, None),
        Token::Minus => (INFIX | PREFIX, "-", Precedence::PSum, None),
        Token::Multiply => (INFIX, "*", Precedence::PProduct, None),
        Token::Divide => (INFIX, "/", Precedence::PProduct, None),
        Token::Power => (INFIX, "^", Precedence::PPower, None),
        Token::Equal => (INFIX, "=", Precedence::PEquals, None),
        Token::NotEqual => (INFIX, "!=", Precedence::PEquals, None),
        Token::GreaterThan => (INFIX, ">", Precedence::PLessGreater, None),
        Token::LessThan => (INFIX, "<", Precedence::PLessGreater, None),
        Token::GreaterThanEqual => (INFIX, ">=", Precedence::PLessGreater, None),
        Token::LessThanEqual => (INFIX, "<=", Precedence::PLessGreater, None),
        Token::BracketL => (OPENING_BRACKET, "(", Precedence::POpening, None),
        Token::BracketR => (CLOSING_BRACKET, ")", Precedence::POpening, None),
        Token::Comma => (SEPARATOR, ",", Precedence::PLowest, None),
        Token::Faculty => (POSTFIX, "!", Precedence::PFaculty, None),
        Token::EOF => (EOF, "", Precedence::PLowest, None),
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

struct Tokens<'a>(&'a [Token]);

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

#[allow(clippy::collapsible_if)]
impl<'a> Iterator for Classifier<'a> {
    type Item = Result<Classification, String>;

    fn next(&mut self) -> Option<Result<Classification, String>> {
        let next = self.next.or_else(|| self.tokens.next());
        self.next = None;

        if let Some(token) = next {
            let token_info = token_info(token);
            if self.expect_operator {
                if token_info.0 & token_type::INFIX != 0 {
                    self.expect_operator = false;
                    create_infix(token_info.1, token_info.2)
                } else if token_info.0 & token_type::POSTFIX != 0 {
                    create_postfix(token_info.1, token_info.2)
                } else if token_info.0 & token_type::EOF != 0 {
                    Some(Ok(Classification::EOF))
                } else if token_info.0 & token_type::CLOSING_BRACKET != 0 {
                    Some(Ok(Classification::Bracket(Bracket {
                        direction: BracketDirection::Closing,
                        r#type: BracketType::Round,
                    })))
                } else if token_info.0 & token_type::SEPARATOR != 0 {
                    self.expect_operator = false;
                    Some(Ok(Classification::Separator))
                } else if token_info.0 & (token_type::IDENT | token_type::OPENING_BRACKET) != 0 {
                    self.expect_operator = false;
                    self.next = Some(token);
                    create_infix("*", Precedence::PProduct)
                } else {
                    Some(Err(format!("Expected operator, found {:?}", token)))
                }
            } else {
                // No operator
                if token_info.0 & token_type::IDENT != 0 {
                    // Is this a function?
                    match token_info.3.expect("Ident in tuple") {
                        Payload::Ident(ident) => {
                            let ident = ident.clone();
                            if self.context.is_function(&ident) {
                                create_function(&ident[..], Precedence::PCall)
                            } else {
                                self.expect_operator = true;
                                Some(Ok(Classification::Ident(ident)))
                            }
                        }
                        _ => panic!("Expected Ident"),
                    }
                } else if token_info.0 & token_type::LITERAL != 0 {
                    self.expect_operator = true;
                    match token_info.3.expect("Value in tuple") {
                        Payload::Number(value) => Some(Ok(Classification::Literal(*value))),
                        _ => panic!("Expected number"),
                    }
                } else if token_info.0 & token_type::PREFIX != 0 {
                    match token {
                        Token::Minus => create_prefix(token_info.1, Precedence::PHighest),
                        _ => create_prefix(token_info.1, token_info.2),
                    }
                } else if token_info.0 & token_type::EOF != 0 {
                    Some(Ok(Classification::EOF))
                } else if token_info.0 & token_type::OPENING_BRACKET != 0 {
                    Some(Ok(Classification::Bracket(Bracket {
                        direction: BracketDirection::Opening,
                        r#type: BracketType::Round,
                    })))
                } else if token_info.0 & token_type::CLOSING_BRACKET != 0 {
                    self.expect_operator = true;
                    Some(Ok(Classification::Bracket(Bracket {
                        direction: BracketDirection::Closing,
                        r#type: BracketType::Round,
                    })))
                } else {
                    Some(Err(format!("Expected literal or ident, found {:?}", token)))
                }
            }
        } else {
            None
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

fn pop_as_symbol(context: &Context, sym_stack: &mut Vec<IdentOrSymbol>) -> Result<Symbol, String> {
    match sym_stack.pop() {
        Some(symbol) => match symbol {
            IdentOrSymbol::Ident(ident) => {
                let fixed = context.is_fixed(&ident);
                Ok(Symbol::new_variable_from_string(ident, fixed))
            }
            IdentOrSymbol::Symbol(symbol) => Ok(symbol),
        },
        None => Err("Expected at least one symbol in sym stack".to_owned()),
    }
}

struct ParseStack {
    pub symbol: Vec<IdentOrSymbol>,
    pub infix: Vec<Operation>,
}

impl ParseStack {
    pub fn pop_as_symbol(&mut self, context: &Context) -> Result<Symbol, String> {
        match self.symbol.pop() {
            Some(symbol) => match symbol {
                IdentOrSymbol::Ident(ident) => {
                    let fixed = context.is_fixed(&ident);
                    Ok(Symbol::new_variable_from_string(ident, fixed))
                }
                IdentOrSymbol::Symbol(symbol) => Ok(symbol),
            },
            None => Err("Expected at least one symbol in parse stack".to_owned()),
        }
    }
}

fn astify(context: &Context, stack: &mut ParseStack, till: Precedence) -> Result<(), String> {
    while !stack.infix.is_empty() && stack.infix.last().expect("infix").precedence > till {
        let Operation { ident, r#type, .. } = stack.infix.pop().unwrap();
        let childs = match r#type {
            OperationType::Infix => {
                let b = stack.pop_as_symbol(context)?;
                let a = stack.pop_as_symbol(context)?;
                vec![a, b] // Order has to be reverted
            }
            OperationType::Prefix => vec![stack.pop_as_symbol(context)?],
            _ => return Err(format!("Invalid argument count {:?}", r#type)),
        };
        stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
            depth: Symbol::calc_depth(&childs),
            flags: context.flags(&ident),
            ident,
            childs,
            value: None,
        }));
    }
    Ok(())
}

fn apply_function(context: &Context, stack: &mut ParseStack) -> Result<(), String> {
    // Create function
    // TODO: Merge this with astify later
    let mut childs = vec![];
    childs.push(pop_as_symbol(context, &mut stack.symbol)?);
    while stack.infix.pop().expect("Something").precedence == Precedence::PSeparator {
        childs.push(pop_as_symbol(context, &mut stack.symbol)?);
    }
    childs.reverse();

    // Was this a function call?
    if !stack.infix.is_empty()
        && stack.infix.last().expect("Some infix").r#type == OperationType::Function
    {
        match stack.infix.pop() {
            Some(func) => stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
                flags: context.flags(&func.ident),
                depth: Symbol::calc_depth(&childs),
                ident: func.ident,
                childs,
                value: None,
            })),
            None => return Err("Some infix".to_owned()),
        }
    } else {
        match childs.pop() {
            Some(child) => stack.symbol.push(IdentOrSymbol::Symbol(child)),
            None => {
                return Err(
                    "Expecting group containing one item. Vectors not supported yet!".to_owned(),
                )
            }
        }
        // assert_eq!(
        //     childs.len(),
        //     1,
        //     "Expecting group containing one item. Vectors not supported yet!"
        // );
        // stack
        //     .symbol
        //     .push(IdentOrSymbol::Symbol(childs.pop().unwrap()));
    }
    Ok(())
}

fn apply_postfix(context: &Context, stack: &mut ParseStack, ident: String) -> Result<(), String> {
    let childs = vec![pop_as_symbol(context, &mut stack.symbol)?];
    stack.symbol.push(IdentOrSymbol::Symbol(Symbol {
        flags: context.flags(&ident),
        depth: Symbol::calc_depth(&childs),
        ident,
        childs,
        value: None,
    }));
    Ok(())
}

pub fn parse(context: &Context, tokens: &[Token]) -> Result<Symbol, String> {
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
                            astify(context, &mut stack, operation.precedence.clone())?;
                        }
                    }
                    stack.infix.push(operation);
                }
                Classification::Prefix(operation) => stack.infix.push(operation),
                Classification::Postfix(operation) => {
                    apply_postfix(context, &mut stack, operation.ident)?
                }
                Classification::Ident(ident) => stack.symbol.push(IdentOrSymbol::Ident(ident)),
                Classification::Literal(value) => stack
                    .symbol
                    .push(IdentOrSymbol::Symbol(Symbol::new_number(value))),
                Classification::EOF => break,
                Classification::Bracket(bracket) => match bracket.direction {
                    BracketDirection::Closing => {
                        astify(context, &mut stack, Precedence::POpening)?;
                        apply_function(context, &mut stack)?;
                    }
                    BracketDirection::Opening => stack.infix.push(Operation {
                        precedence: Precedence::POpening,
                        ident: String::from("("),
                        r#type: OperationType::Dummy,
                    }),
                },
                Classification::Separator => {
                    stack.infix.push(Operation {
                        precedence: Precedence::PSeparator,
                        ident: String::from(","),
                        r#type: OperationType::Dummy,
                    });
                }
            },
            Err(err) => return Err(err),
        };
    }

    astify(context, &mut stack, Precedence::PLowest)?;
    assert!(stack.infix.is_empty());
    assert_eq!(stack.symbol.len(), 1);
    pop_as_symbol(context, &mut stack.symbol)
}

#[cfg(test)]
mod specs {
    use super::*;
    use std::collections::HashMap;
    use test::Bencher;

    fn new_variable(ident: &str) -> Symbol {
        Symbol::new_variable(ident, false)
    }

    fn new_op(ident: &str, childs: Vec<Symbol>) -> Symbol {
        Symbol::new_operator(ident, false, false, childs)
    }

    fn new_func(ident: &str, childs: Vec<Symbol>) -> Symbol {
        Symbol::new_operator(ident, false, false, childs)
    }

    fn create_function(ident: &str) -> Classification {
        Classification::Prefix(Operation {
            precedence: Precedence::PCall,
            ident: String::from(ident),
            r#type: OperationType::Function,
        })
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
    fn classifier_single_ident_no_args() {
        let context = Context {
            declarations: HashMap::new(),
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
            declarations: HashMap::new(),
        };
        let tokens = vec![Token::Ident(String::from("a")), Token::EOF];
        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(actual, new_variable("a"));
    }

    #[test]
    fn classifier_function_with_single_arg() {
        let context = create_context(vec!["f"]);

        let raw_tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::BracketR,
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
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Opening,
                    r#type: BracketType::Round
                })),
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Closing,
                    r#type: BracketType::Round
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
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::BracketR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(
            actual,
            Symbol::new_operator("f", false, false, vec![new_variable("a")])
        );
    }

    #[test]
    fn classifier_function_with_multiple_args() {
        let context = create_context(vec!["f"]);
        let raw_tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Comma,
            Token::Ident(String::from("b")),
            Token::Comma,
            Token::Ident(String::from("c")),
            Token::BracketR,
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
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Opening,
                    r#type: BracketType::Round
                })),
                Ok(Classification::Ident(String::from("a"))),
                Ok(Classification::Separator),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::Separator),
                Ok(Classification::Ident(String::from("c"))),
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Closing,
                    r#type: BracketType::Round
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
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Comma,
            Token::Ident(String::from("b")),
            Token::Comma,
            Token::Ident(String::from("c")),
            Token::BracketR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");

        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                false,
                false,
                vec![new_variable("a"), new_variable("b"), new_variable("c")]
            )
        );
    }

    #[test]
    fn function_nested_simple() {
        let context = create_context(vec!["f", "g"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::Ident(String::from("g")),
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::BracketR,
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                false,
                false,
                vec![Symbol::new_operator(
                    "g",
                    false,
                    false,
                    vec![new_variable("a")]
                )]
            )
        );
    }

    #[test]
    fn function_inner_operator() {
        // f(a + b)
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(
            actual,
            Symbol::new_operator(
                "f",
                false,
                false,
                vec![Symbol::new_operator(
                    "+",
                    false,
                    false,
                    vec![new_variable("a"), new_variable("b"),]
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
            Token::BracketL,
            Token::Ident(String::from("g")),
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::BracketR,
            Token::Plus,
            Token::Ident(String::from("h")),
            Token::BracketL,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(
            actual,
            new_op(
                "f",
                vec![new_op(
                    "+",
                    vec![
                        new_op("g", vec![new_variable("a")]),
                        new_op("h", vec![new_variable("b"),])
                    ]
                )]
            )
        );

        b.iter(|| {
            parse(&context, &tokens).unwrap();
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
        let actual = parse(&context, &tokens).expect("parse");
        assert_eq!(
            actual,
            new_op("+", vec![new_variable("a"), new_variable("b")])
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
        let actual = parse(&context, &tokens).expect("parse");

        assert_eq!(
            actual,
            new_op(
                "+",
                vec![
                    new_variable("a"),
                    new_op(
                        "-",
                        vec![
                            new_op("*", vec![new_variable("b"), new_variable("c")]),
                            new_variable("d")
                        ]
                    ),
                ]
            )
        );
    }

    #[test]
    fn operators_with_brackets_front() {
        // (a+b)*c
        let context = create_context(vec![]);
        let tokens = vec![
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "*",
            vec![
                new_op("+", vec![new_variable("a"), new_variable("b")]),
                new_variable("c"),
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
            Token::BracketL,
            Token::Ident(String::from("b")),
            Token::BracketR,
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
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Opening,
                    r#type: BracketType::Round
                })),
                Ok(Classification::Ident(String::from("b"))),
                Ok(Classification::Bracket(Bracket {
                    direction: BracketDirection::Closing,
                    r#type: BracketType::Round
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
            Token::BracketL,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "+",
            vec![new_variable("a"), new_op("f", vec![new_variable("b")])],
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
                    precedence: Precedence::PHighest,
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

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op("-", vec![new_variable("a")]);
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

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op("*", vec![new_variable("a"), new_variable("b")]);
        assert_eq!(actual, expected);
    }

    #[bench]
    fn implicit_bin_operator_brackets(b: &mut Bencher) {
        // ab -> (a+b)(c+d)*e(f+g)
        let context = create_context(vec![]);
        let tokens = vec![
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::BracketL,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::Ident(String::from("d")),
            Token::BracketR,
            Token::Multiply,
            Token::Ident(String::from("e")),
            Token::BracketL,
            Token::Ident(String::from("f")),
            Token::Plus,
            Token::Ident(String::from("g")),
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "*",
            vec![
                new_op("+", vec![new_variable("a"), new_variable("b")]),
                new_op(
                    "*",
                    vec![
                        new_op("+", vec![new_variable("c"), new_variable("d")]),
                        new_op(
                            "*",
                            vec![
                                new_variable("e"),
                                new_op("+", vec![new_variable("f"), new_variable("g")]),
                            ],
                        ),
                    ],
                ),
            ],
        );
        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens).unwrap();
        })
    }

    #[bench]
    fn function_complex_inner(b: &mut Bencher) {
        // f((a+b)*c+d*(e+h))
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::Multiply,
            Token::Ident(String::from("c")),
            Token::Plus,
            Token::Ident(String::from("d")),
            Token::Multiply,
            Token::BracketL,
            Token::Ident(String::from("e")),
            Token::Plus,
            Token::Ident(String::from("h")),
            Token::BracketR,
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");

        let expected = new_func(
            "f",
            vec![new_op(
                "+",
                vec![
                    new_op(
                        "*",
                        vec![
                            new_op("+", vec![new_variable("a"), new_variable("b")]),
                            new_variable("c"),
                        ],
                    ),
                    new_op(
                        "*",
                        vec![
                            new_variable("d"),
                            new_op("+", vec![new_variable("e"), new_variable("h")]),
                        ],
                    ),
                ],
            )],
        );

        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens).unwrap();
        })
    }

    #[test]
    fn double_brackets() {
        // ((a+b))
        let context = create_context(vec![]);
        let tokens = vec![
            Token::BracketL,
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Ident(String::from("b")),
            Token::BracketR,
            Token::BracketR,
            Token::EOF,
        ];

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op("+", vec![new_variable("a"), new_variable("b")]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn function_double_brackets() {
        // f((a))
        let context = create_context(vec!["f"]);
        let tokens = vec![
            Token::Ident(String::from("f")),
            Token::BracketL,
            Token::BracketL,
            Token::Ident(String::from("a")),
            Token::BracketR,
            Token::BracketR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_func("f", vec![new_variable("a")]);
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

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "*",
            vec![new_variable("a"), new_op("-", vec![new_variable("b")])],
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn postfix_operator_simple() {
        // a!
        let tokens = vec![Token::Ident(String::from("a")), Token::Faculty, Token::EOF];
        let context = create_context(vec![]);

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op("!", vec![new_variable("a")]);

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
            Token::BracketL,
            Token::Ident(String::from("e")),
            Token::Multiply,
            Token::Ident(String::from("d")),
            Token::BracketR,
            Token::Faculty,
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "+",
            vec![
                new_variable("a"),
                new_op(
                    "+",
                    vec![
                        new_op(
                            "*",
                            vec![new_op("!", vec![new_variable("b")]), new_variable("c")],
                        ),
                        new_op(
                            "!",
                            vec![new_op("*", vec![new_variable("e"), new_variable("d")])],
                        ),
                    ],
                ),
            ],
        );

        assert_eq!(actual, expected);
        b.iter(|| {
            parse(&context, &tokens).unwrap();
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

        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "+",
            vec![
                new_op(
                    "*",
                    vec![
                        new_variable("a"),
                        new_op("^", vec![new_variable("b"), new_variable("c")]),
                    ],
                ),
                new_op(
                    "*",
                    vec![
                        new_op("^", vec![new_variable("d"), new_variable("e")]),
                        new_variable("f"),
                    ],
                ),
            ],
        );

        assert_eq!(actual, expected);

        b.iter(|| {
            parse(&context, &tokens).unwrap();
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

        let actual = parse(&context, &tokens).expect("parse");

        let expected = new_op(
            "+",
            vec![
                Symbol::new_number(1),
                new_op("*", vec![Symbol::new_number(2), Symbol::new_number(3)]),
            ],
        );

        assert_eq!(actual, expected);
    }

    // Regressions

    #[test]
    fn equation_1() {
        // a - b = 0
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Minus,
            Token::Ident(String::from("b")),
            Token::Equal,
            Token::Number(0),
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens).expect("parse");

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
    fn equation_3() {
        // x = -a
        let tokens = vec![
            Token::Ident(String::from("x")),
            Token::Equal,
            Token::Minus,
            Token::Ident(String::from("a")),
            Token::EOF,
        ];

        let context = create_context(vec![]);

        let actual = parse(&context, &tokens).expect("parse");

        let expected = new_op(
            "=",
            vec![new_variable("x"), new_op("-", vec![new_variable("a")])],
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn non_associative() {
        // a-(b+c)
        let context = create_context(vec![]);
        let tokens = vec![
            Token::Ident(String::from("a")),
            Token::Minus,
            Token::BracketL,
            Token::Ident(String::from("b")),
            Token::Plus,
            Token::Ident(String::from("c")),
            Token::BracketR,
            Token::EOF,
        ];
        let actual = parse(&context, &tokens).expect("parse");
        let expected = new_op(
            "-",
            vec![
                new_variable("a"),
                new_op("+", vec![new_variable("b"), new_variable("c")]),
            ],
        );
        assert_eq!(actual.to_string(), expected.to_string());
        assert_eq!(actual, expected);
    }

    #[test]
    fn bug_unary_minus() {
        // -a + 0
        let tokens = vec![
            Token::Minus,
            Token::Ident(String::from("a")),
            Token::Plus,
            Token::Number(0),
            Token::EOF,
        ];

        let context = Context::standard();

        let actual = parse(&context, &tokens).expect("parse");

        let expected = Symbol::new_operator(
            "+",
            true,
            false,
            vec![
                Symbol::new_operator("-", true, false, vec![new_variable("a")]),
                Symbol::new_number(0),
            ],
        );

        assert_eq!(actual, expected);
    }
}
