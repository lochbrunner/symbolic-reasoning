use super::base::*;
use crate::common::RefEquality;
use crate::parser::Precedence;
use crate::Symbol;
use std::collections::HashMap;
use std::fmt;

fn create_context<'a>(decoration: &'a [Decoration], verbose: bool) -> FormatContext<'a> {
    FormatContext {
        operators: Operators {
            infix: hashmap! {
                "+" => Precedence::PSum,
                "-" => Precedence::PSum,
                "*" => Precedence::PProduct,
                "/" => Precedence::PProduct,
                "^" => Precedence::PPower,
                "=" => Precedence::PEquals,
                "==" => Precedence::PEquals,
                "!=" => Precedence::PEquals,
            },
            postfix: hashset! {"!"},
            prefix: hashset! {"-"},
            non_associative: if verbose {
                hashset! {"-","/", "+", "*", "^"}
            } else {
                hashset! {"-","/"}
            },
        },
        formats: SpecialFormatRules {
            symbols: hashmap! {},
            functions: hashmap! {},
        },
        decoration,
    }
}

pub fn dump_plain(symbol: &Symbol, decoration: &[Decoration], verbose: bool) -> String {
    let context = create_context(decoration, verbose);
    let mut string = String::new();
    dump_base(
        &context,
        symbol,
        FormattingLocation::new(),
        true,
        &mut NoOpDumpingHood {},
        &mut string,
    );
    string
}

fn path_to_str(path: &Vec<usize>) -> String {
    path.iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn dump_plain_with_path(
    symbol: &Symbol,
    decoration: &[Decoration],
    verbose: bool,
) -> (String, HashMap<String, (usize, usize)>) {
    let context = create_context(decoration, verbose);
    let mut recorder_hook = RecorderDumpingHook::default();
    let mut string = String::new();
    dump_base(
        &context,
        symbol,
        FormattingLocation::new(),
        true,
        &mut recorder_hook,
        &mut string,
    );
    let mut path_to_span = HashMap::<String, (usize, usize)>::new();
    for (path, child) in symbol.iter_dfs_path() {
        let begin = recorder_hook.begin_positions[&RefEquality(child)];
        let end = recorder_hook.end_positions[&RefEquality(child)];
        path_to_span.insert(path_to_str(&path), (begin, end));
    }
    (string, path_to_span)
}

pub fn dump_plain_with_bfs_pos(
    symbol: &Symbol,
    decoration: &[Decoration],
    verbose: bool,
) -> (String, Vec<(usize, usize)>) {
    let (string, path_to_span) = dump_plain_with_path(symbol, decoration, verbose);
    let paths: Vec<_> = symbol
        .iter_bfs_path()
        .map(|(path, _)| path_to_str(&path))
        .collect();
    // (string, path_to_span)
    (
        string,
        paths.into_iter().map(|path| path_to_span[&path]).collect(),
    )
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", dump_plain(self, &[], false))
    }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::context::*;
    use std::collections::HashMap;

    fn create_context(function_names: &[&str]) -> Context {
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

    fn test(code: &str) {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, code).unwrap();
        assert_eq!(dump_plain(&term, &[], false), String::from(code));
    }

    fn test_verbose(code: &str) {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, code).unwrap();
        assert_eq!(dump_plain(&term, &[], true), String::from(code));
    }

    fn test_with_function(function_names: &[&str], code: &str) {
        let context = create_context(function_names);
        let term = Symbol::parse(&context, code).unwrap();
        assert_eq!(dump_plain(&term, &[], false), String::from(code));
    }

    #[test]
    fn function_simple() {
        test_with_function(&["f"], "f(a)");
    }

    #[test]
    fn infix_simple() {
        test("a+b");
    }

    #[test]
    fn infix_precedence() {
        test("a+b*c");
    }

    #[test]
    fn infix_parenthesis() {
        test("(a+b)*c");
    }

    #[test]
    fn postfix_simple() {
        test("a!");
    }

    #[test]
    fn postfix_with_infix() {
        test("(a+b)!");
    }

    #[test]
    fn verbose_infix_simple() {
        test_verbose("a+(b+c)")
    }

    #[test]
    fn unary_minus() {
        test("-a");
    }

    #[test]
    fn unary_minus_multiplication() {
        test("-a*b");
    }

    #[test]
    fn unary_minus_multiplication_dominant() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "-(a*b)").unwrap();
        assert_eq!(dump_plain(&term, &[], false), String::from("-1*a*b"));
    }

    #[test]
    fn dump_with_path() {
        let context = Context::standard();
        let term = Symbol::parse(&context, "x+1=a-1").unwrap();
        let (string, mapping) = dump_plain_with_path(&term, &[], false);
        assert_eq!(&string, "x+1=a-1");
        let expected_mapping = hashmap! {
        "0/0"=> (0,1),
        "0"=> (1,2),
        "0/1"=> (2,3),
        ""=> (3, 4),
        "1/0"=> (4,5),
        "1"=> (5,6),
        "1/1"=> (6,7),
        };
        let expected_mapping = expected_mapping
            .into_iter()
            .map(|(k, (b, e))| (k.to_string(), (b as usize, e as usize)))
            .collect();
        assert_eq!(mapping, expected_mapping);
    }

    #[test]
    fn dump_with_bfs_pos() {
        let context = Context::standard();
        let term = Symbol::parse(&context, "x+1=a-1").unwrap();
        let (string, spans) = dump_plain_with_bfs_pos(&term, &[], false);
        assert_eq!(&string, "x+1=a-1");
        let expected_spans = vec![(3, 4), (1, 2), (5, 6), (0, 1), (2, 3), (4, 5), (6, 7)];
        let expected_spans: Vec<_> = expected_spans
            .into_iter()
            .map(|(b, e)| (b as usize, e as usize))
            .collect();
        assert_eq!(spans, expected_spans);
    }

    #[test]
    fn bug_15_subtraction() {
        test("a-(b+c)");
    }

    #[test]
    fn bug_15_subtraction_false_positive() {
        test("a-b");
    }

    #[test]
    fn bug_16() {
        test("(a+b)^c");
    }
}
