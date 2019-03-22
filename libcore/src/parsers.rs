use super::rule;
use super::symbol;

named!(
    list<&str,Vec<symbol::Symbol>>,
    ws!(delimited!(
        char!('('),
        separated_list!(char!(','), parse_symbol_new),
        char!(')')
    ))
);

fn is_alphabetic_c(chr: char) -> bool {
    (chr as u8 >= 0x41 && chr as u8 <= 0x5A) || (chr as u8 >= 0x61 && chr as u8 <= 0x7A)
}

named!(
    string<&str, &str>,
    escaped!(take_while1!(is_alphabetic_c), '\\', one_of!("\"n\\"))
);

named!(
    parse_symbol_new<&str,symbol::Symbol>,
    ws!(alt!(
      parse_symbol_with_childs => {|o| o} |
      string =>   {|s| symbol::Symbol::new_variable(s)}
    ))
);



named!(
    parse_symbol_with_childs<&str,symbol::Symbol>,
    do_parse!(
        ident: string
            >> childs: list
            >> (symbol::Symbol {
                ident: String::from(ident),
                depth: symbol::Symbol::calc_depth(&childs),
                childs,
                fixed: ident.chars().nth(0).unwrap().is_uppercase(),
                //fixed: true
            })
    )  
);

named!(
    pub parse_rule<&str, rule::Rule>,
    ws!(
        do_parse!(
            condition: parse_symbol_new >>
            tag!("=>") >>
            conclusion: parse_symbol_new >>
            (rule::Rule{condition, conclusion})
        )
    )
);

impl symbol::Symbol {
    pub fn parse(s: &str) -> symbol::Symbol {
        let p = parse_symbol_new(s);
        p.unwrap().1
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn parse_symbol() {
        assert_eq!(
            symbol::Symbol::parse("A(a,b,c)\0"),
            symbol::Symbol {
                ident: String::from("A"),
                depth: 2,
                fixed: true,
                childs: vec![
                    symbol::Symbol::new_variable("a"),
                    symbol::Symbol::new_variable("b"),
                    symbol::Symbol::new_variable("c"),
                ]
            }
        );
    }

    #[test]
    fn fixed_variable() {
        let v = symbol::Symbol::parse("A\0");
        assert!(v.fixed);
    }

    #[test]
    fn non_fixed_variable() {
        let v = symbol::Symbol::parse("a\0");
        assert!(!v.fixed);
    }

    #[test]
    fn operator_e2e() {
        let op = symbol::Symbol::parse("A(a,b,c)\0");
        assert_eq!(op.to_string(), "A(a,b,c)");
        assert!(op.fixed);
        assert_eq!(op.depth, 2);
    }

    #[test]
    fn operator_depth() {
        let op = symbol::Symbol::parse("A(B(c),b,c)\0");
        assert!(op.fixed);
        assert_eq!(op.depth, 3);
    }

    #[test]
    fn operator_single_e2e() {
        let op = symbol::Symbol::parse("A(a)\0");
        assert_eq!(op.to_string(), "A(a)");
    }

    #[test]
    fn operator_nested_e2e() {
        let op = symbol::Symbol::parse("A(B(e),b,c)\0");
        assert_eq!(op.to_string(), "A(B(e),b,c)");
    }

    #[test]
    fn rule_e2e_variable() {
        let rule = rule::Rule::parse("a  => a\0");
        assert_eq!(rule.to_string(), "a => a")
    }

    #[test]
    fn rule_e2e_operator() {
        let rule = rule::Rule::parse("A(a,b)  => B(c,d)\0");
        assert_eq!(rule.to_string(), "A(a,b) => B(c,d)")
    }
}
