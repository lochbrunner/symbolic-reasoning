use super::base::*;
use crate::parser::Precedence;
use crate::{Rule, Symbol};

pub trait LaTeX {
    fn write_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write;

    fn writeln_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write;
}

pub fn dump_latex(symbol: &Symbol, decoration: &[Decoration], verbose: bool) -> String {
    let context = FormatContext {
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
            symbols: hashmap! {"(" => "\\left( ", ")" => "\\right) ", "*" => "\\cdot ", "!=" => "\\neq ", "==" => "="},
            functions: hashmap! {
                "^" => vec![
                    FormatItem::Tag("{"),
                    FormatItem::Child(0, false),
                    FormatItem::Tag("}^{"),
                    FormatItem::Child(1, true),
                    FormatItem::Tag("}"),
                ],
                "/" => vec![
                    FormatItem::Tag("\\frac{"),
                    FormatItem::Child(0, true),
                    FormatItem::Tag("}{"),
                    FormatItem::Child(1, true),
                    FormatItem::Tag("}"),
                ],
                "root" => vec![
                    FormatItem::Tag("\\sqrt["),
                    FormatItem::Child(0, true),
                    FormatItem::Tag("]{"),
                    FormatItem::Child(1, true),
                    FormatItem::Tag("}"),
                ],
                "sqrt" => vec![
                    FormatItem::Tag("\\sqrt{"),
                    FormatItem::Child(0, true),
                    FormatItem::Tag("}"),
                ],
                "D" => vec![
                    FormatItem::Tag("\\frac{\\partial}{\\partial "),
                    FormatItem::Child(1, true),
                    FormatItem::Tag("}"),
                    FormatItem::Child(0, true),
                ]
            },
        },
        decoration,
    };
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

impl Symbol {
    pub fn write_latex_highlight<W>(
        &self,
        path: &[usize],
        writer: &mut W,
    ) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        writer.write(
            dump_latex(
                self,
                &[Decoration {
                    path,
                    pre: "\\mathbin{\\textcolor{red}{",
                    post: "}}",
                }],
                false,
            )
            .as_bytes(),
        )?;
        Ok(())
    }
}

impl LaTeX for Symbol {
    fn write_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        write!(writer, "{}", dump_latex(self, &[], false))
    }

    fn writeln_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        writeln!(writer, "{}", dump_latex(self, &[], false))
    }
}

impl LaTeX for Rule {
    fn write_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        write!(
            writer,
            "{} \\Rightarrow {}",
            dump_latex(&self.condition, &[], false),
            dump_latex(&self.conclusion, &[], false)
        )
    }

    fn writeln_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        writeln!(
            writer,
            "{} \\Rightarrow {}",
            dump_latex(&self.condition, &[], false),
            dump_latex(&self.conclusion, &[], false)
        )
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

    #[test]
    fn fraction_simple() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a/b").unwrap();
        assert_eq!(dump_latex(&term, &[], false), String::from("\\frac{a}{b}"));
    }
    #[test]
    fn inequality() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a!=b").unwrap();
        assert_eq!(dump_latex(&term, &[], false), String::from("a\\neq b"));
    }
    #[test]
    fn fraction_double() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a/(b/c)").unwrap();
        assert_eq!(
            dump_latex(&term, &[], false),
            String::from("\\frac{a}{\\frac{b}{c}}")
        );
    }

    #[test]
    fn sqrt_simple() {
        let context = create_context(&["sqrt"]);
        let term = Symbol::parse(&context, "sqrt(a)").unwrap();
        assert_eq!(dump_latex(&term, &[], false), String::from("\\sqrt{a}"));
    }

    #[test]
    fn root_simple() {
        let context = create_context(&["root"]);
        let term = Symbol::parse(&context, "root(a+b,3+1)").unwrap();
        assert_eq!(
            dump_latex(&term, &[], false),
            String::from("\\sqrt[a+b]{3+1}")
        );
    }

    #[test]
    fn derivative_simple() {
        let context = create_context(&["D", "f"]);
        let term = Symbol::parse(&context, "D(f(x),x)").unwrap();
        assert_eq!(
            dump_latex(&term, &[], false),
            String::from("\\frac{\\partial}{\\partial x}f\\left( x\\right) ")
        );
    }

    #[test]
    fn brackets() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a*(b+c)").unwrap();
        assert_eq!(
            dump_latex(&term, &[], false),
            String::from("a\\cdot \\left( b+c\\right) ")
        );
    }

    #[test]
    fn double_super_script_outer() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a^b^c").unwrap();
        assert_eq!(dump_latex(&term, &[], false), String::from("{{a}^{b}}^{c}"));
    }

    #[test]
    fn double_super_script_inner() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "(a^b)^c").unwrap();
        assert_eq!(dump_latex(&term, &[], false), String::from("{{a}^{b}}^{c}"));
    }

    #[test]
    fn bug_16() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "(a+b)^c").unwrap();
        assert_eq!(
            dump_latex(&term, &[], false),
            String::from("{\\left( a+b\\right) }^{c}")
        );
    }

    #[test]
    fn decoration_variable() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a").unwrap();
        let path = vec![];
        let deco = [Decoration {
            path: &path,
            pre: "<",
            post: ">",
        }];

        assert_eq!(dump_latex(&term, &deco, false), String::from("<a>"));
    }

    #[test]
    fn decoration_operator() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a+b").unwrap();
        let path = vec![0];
        let deco = [Decoration {
            path: &path,
            pre: "<",
            post: ">",
        }];

        assert_eq!(dump_latex(&term, &deco, false), String::from("<a>+b"));
    }

    #[test]
    fn decoration_all() {
        let context = create_context(&[]);
        let term = Symbol::parse(&context, "a+b").unwrap();
        let path_0 = vec![0];
        let path_r = vec![];
        let path_1 = vec![1];
        let deco = [
            Decoration {
                path: &path_0,
                pre: "<A>",
                post: "</A>",
            },
            Decoration {
                path: &path_1,
                pre: "<B>",
                post: "</B>",
            },
            Decoration {
                path: &path_r,
                pre: "<C>",
                post: "</C>",
            },
        ];

        assert_eq!(
            dump_latex(&term, &deco, false),
            String::from("<C><A>a</A>+<B>b</B></C>")
        );
    }
}
