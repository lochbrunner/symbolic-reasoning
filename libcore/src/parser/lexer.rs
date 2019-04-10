use nom::types::*;
use nom::*;
use std::str;
use std::str::Utf8Error;

// pub mod token;
use super::token::Token;

// operators
named!(equal_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("==") >> (Token::Equal))
);

named!(not_equal_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("!=") >> (Token::NotEqual))
);

named!(plus_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("+") >> (Token::Plus))
);

named!(minus_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("-") >> (Token::Minus))
);

named!(multiply_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("*") >> (Token::Multiply))
);

named!(power_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("^") >> (Token::Power))
);

named!(divide_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("/") >> (Token::Divide))
);

named!(faculty_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("!") >> (Token::Faculty))
);

named!(greater_operator_equal<CompleteByteSlice, Token>,
  do_parse!(tag!(">=") >> (Token::GreaterThanEqual))
);

named!(lesser_operator_equal<CompleteByteSlice, Token>,
  do_parse!(tag!("<=") >> (Token::LessThanEqual))
);

named!(greater_operator<CompleteByteSlice, Token>,
  do_parse!(tag!(">") >> (Token::GreaterThan))
);

named!(lesser_operator<CompleteByteSlice, Token>,
  do_parse!(tag!("<") >> (Token::LessThan))
);

named!(lex_operator<CompleteByteSlice, Token>, alt!(
    equal_operator |
    not_equal_operator |
    plus_operator |
    minus_operator |
    multiply_operator |
    divide_operator |
    power_operator |
    faculty_operator |
    greater_operator_equal |
    lesser_operator_equal |
    greater_operator |
    lesser_operator
));

// Macros
macro_rules! check(
  ($input:expr, $submac:ident!( $($args:tt)* )) => (
    {
      use std::result::Result::*;
      use nom::{Err,ErrorKind};

      let mut failed = false;
      for &idx in $input.0 {
        if !$submac!(idx, $($args)*) {
            failed = true;
            break;
        }
      }
      if failed {
        let e: ErrorKind<u32> = ErrorKind::Tag;
        Err(Err::Error(error_position!($input, e)))
      } else {
        Ok((&b""[..], $input))
      }
    }
  );
  ($input:expr, $f:expr) => (
    check!($input, call!($f));
  );
);

// Reserved or ident
fn parse_ident(c: CompleteStr, rest: Option<CompleteStr>) -> Token {
  let mut string = c.0.to_owned();
  string.push_str(rest.unwrap_or(CompleteStr("")).0);
  // match string.as_ref {
  //     _ => Token::Ident(string),
  // }
  Token::Ident(string)
}

fn complete_byte_slice_str_from_utf8(c: CompleteByteSlice) -> Result<CompleteStr, Utf8Error> {
  str::from_utf8(c.0).map(|s| CompleteStr(s))
}

named!(take_1_char<CompleteByteSlice, CompleteByteSlice>,
    flat_map!(take!(1), check!(is_alphabetic))
);

named!(lex_ident<CompleteByteSlice, Token>,
    do_parse!(
        c: map_res!(call!(take_1_char), complete_byte_slice_str_from_utf8) >>
        rest: opt!(complete!(map_res!(alphanumeric, complete_byte_slice_str_from_utf8))) >>
        (parse_ident(c, rest))
    )
);

// punctuations
named!(comma<CompleteByteSlice, Token>,
  do_parse!(tag!(",") >> (Token::Comma))
);
named!(lparen<CompleteByteSlice, Token>,
  do_parse!(tag!("(") >> (Token::ParenL))
);

named!(rparen<CompleteByteSlice, Token>,
  do_parse!(tag!(")") >> (Token::ParenR))
);

named!(lex_punctuations<CompleteByteSlice, Token>, alt!(
    comma |
    lparen |
    rparen
));

named!(lex_token<CompleteByteSlice, Token>, alt_complete!(
    lex_operator |
    lex_punctuations |
    lex_ident
    // lex_integer |
    // lex_illegal
));

named!(lex_tokens_mac<CompleteByteSlice, Vec<Token>>, ws!(many0!(lex_token)));

pub fn lex_tokens(bytes: &[u8]) -> IResult<CompleteByteSlice, Vec<Token>> {
  lex_tokens_mac(CompleteByteSlice(bytes))
    .map(|(slice, result)| (slice, [&result[..], &vec![Token::EOF][..]].concat()))
}

#[cfg(test)]
mod specs {
  use super::*;

  #[test]
  fn ident_mixed() {
    let input = " abc de,a,bc,, (a,b)".as_bytes();
    let (_, actual) = lex_tokens(input).unwrap();
    let expected = vec![
      Token::Ident("abc".to_owned()),
      Token::Ident("de".to_owned()),
      Token::Comma,
      Token::Ident("a".to_owned()),
      Token::Comma,
      Token::Ident("bc".to_owned()),
      Token::Comma,
      Token::Comma,
      Token::ParenL,
      Token::Ident("a".to_owned()),
      Token::Comma,
      Token::Ident("b".to_owned()),
      Token::ParenR,
      Token::EOF,
    ];

    assert_eq!(actual, expected);
  }

  #[test]
  fn operator() {
    let input = "a+b*c^d-e/f".as_bytes();
    let (_, actual) = lex_tokens(input).unwrap();
    let expected = vec![
      Token::Ident("a".to_owned()),
      Token::Plus,
      Token::Ident("b".to_owned()),
      Token::Multiply,
      Token::Ident("c".to_owned()),
      Token::Power,
      Token::Ident("d".to_owned()),
      Token::Minus,
      Token::Ident("e".to_owned()),
      Token::Divide,
      Token::Ident("f".to_owned()),
      Token::EOF,
    ];

    assert_eq!(actual, expected);
  }
}
