use super::rule;
use super::symbol;

named!(
    array<&str,Vec<symbol::Symbol>>,
    ws!(delimited!(
        char!('('),
        separated_list!(char!(','), parse_symbol),
        char!(')')
    ))
);

named!(
    operator<&str,symbol::Operator>,
    do_parse!(
        ident: string
            >> childs: array
            >> (symbol::Operator {
                ident: String::from(ident),
                childs
            })
    )
);

fn is_alphabetic_c(chr: char) -> bool {
    (chr as u8 >= 0x41 && chr as u8 <= 0x5A) || (chr as u8 >= 0x61 && chr as u8 <= 0x7A)
}

named!(
    string<&str, &str>,
        escaped!(take_while1!(is_alphabetic_c), '\\', one_of!("\"n\\"))
);

named!(
    pub parse_symbol<&str,symbol::Symbol>,
    ws!(alt!(
      operator => {|o| symbol::Symbol::Operator(o)} |
      string =>   {|s| symbol::Symbol::Variable(symbol::Variable{ident: String::from(s)}) }
    ))
);

named!(
    pub parse_rule<&str, rule::Rule>,
    ws!(
        do_parse!(
            condition: parse_symbol >>
            tag!("=>") >>
            conclusion: parse_symbol >>
            (rule::Rule{condition, conclusion})
        )
    )
);

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn parse_symbol() {
        assert_eq!(
            symbol::Symbol::new("A(a,b,c)\0"),
            symbol::Symbol::Operator(symbol::Operator {
                ident: String::from("A"),
                childs: vec![
                    symbol::Symbol::Variable(symbol::Variable {
                        ident: String::from("a")
                    }),
                    symbol::Symbol::Variable(symbol::Variable {
                        ident: String::from("b")
                    }),
                    symbol::Symbol::Variable(symbol::Variable {
                        ident: String::from("c")
                    })
                ]
            })
        );

        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn operator_e2e() {
        let op = symbol::Symbol::new("A(a,b,c)\0");
        assert_eq!(op.to_string(), "A(a,b,c)");
    }

    #[test]
    fn operator_singe_e2e() {
        let op = symbol::Symbol::new("A(a)\0");
        assert_eq!(op.to_string(), "A(a)");
    }

    #[test]
    fn operator_nested_e2e() {
        let op = symbol::Symbol::new("A(B(e),b,c)\0");
        assert_eq!(op.to_string(), "A(B(e),b,c)");
    }

    #[test]
    fn rule_e2e_variable() {
        let rule = rule::Rule::new("a  => a\0");
        assert_eq!(rule.to_string(), "a => a")
    }

    #[test]
    fn rule_e2e_operator() {
        let rule = rule::Rule::new("A(a,b)  => B(c,d)\0");
        assert_eq!(rule.to_string(), "A(a,b) => B(c,d)")
    }
}
