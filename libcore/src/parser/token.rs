#[derive(PartialEq, Debug, Clone)]
pub enum Token {
    EOF,
    //
    Ident(String),
    Number(i64),
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
    BracketL,
    BracketR,
}
