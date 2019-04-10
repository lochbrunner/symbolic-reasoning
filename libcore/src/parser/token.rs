#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    EOF,
    //
    Ident(String),
    // Number(f64),
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
    Faculty,
    // grouping
    Comma,
    ParenL,
    ParenR,
}
