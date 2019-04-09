#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    Illegal,
    EOF,
    //
    Ident(String),
    Number(f64),
    // operators
    Plus,
    Minus,
    Divide,
    Multiply,
    Power,
    Equal,
    NotEqual,
    GreaterThanEqual,
    LessThanEqual,
    GreaterThan,
    LessThan,
    // Not,
    Faculty,
    // grouping
    Comma,
    ParenL,
    ParenR,
}
