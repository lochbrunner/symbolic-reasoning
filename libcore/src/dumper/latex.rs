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

pub fn dump_latex(symbol: &Symbol, embedding: &Option<Decoration>) -> String {
    let special_symbols = SpecialSymbols {
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
        format: SpecialFormatRules {
            symbols: hashmap! {"(" => "\\left ", ")" => "\\right "},
            functions: hashmap! {
                "^" => vec![
                    FormatItem::Child(0),
                    FormatItem::Tag("^{"),
                    FormatItem::Child(1),
                    FormatItem::Tag("}"),
                ],
                "/" => vec![
                    FormatItem::Tag("\\frac{"),
                    FormatItem::Child(0),
                    FormatItem::Tag("}{"),
                    FormatItem::Child(1),
                    FormatItem::Tag("}"),
                ]
            },
        },
    };
    let mut string = String::new();
    dump_base(
        &special_symbols,
        symbol,
        &mut string,
        embedding,
        FormatingLocation::new(),
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
        write!(
            writer,
            "{}",
            dump_latex(
                self,
                &Some(Decoration {
                    path,
                    pre: "\\mathbin{\\textcolor{red}{",
                    post: "}}",
                })
            )
        )
    }
}

impl LaTeX for Symbol {
    fn write_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        write!(writer, "{}", dump_latex(self, &None))
    }

    fn writeln_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        write!(writer, "{}\n", dump_latex(self, &None))
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
            dump_latex(&self.condition, &None),
            dump_latex(&self.conclusion, &None)
        )
    }

    fn writeln_latex<W>(&self, writer: &mut W) -> Result<(), std::io::Error>
    where
        W: std::io::Write,
    {
        writeln!(
            writer,
            "{} \\Rightarrow {}",
            dump_latex(&self.condition, &None),
            dump_latex(&self.conclusion, &None)
        )
    }
}

#[cfg(test)]
mod e2e {
    use super::*;
    use crate::context::*;
    use std::collections::HashMap;

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
    fn fraction_simple() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a/b");
        assert_eq!(dump_latex(&term, &None), String::from("\\frac{a}{b}"));
    }

    #[test]
    fn fraction_double() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a/(b/c)");
        assert_eq!(
            dump_latex(&term, &None),
            String::from("\\frac{a}{\\frac{b}{c}}")
        );
    }

    #[test]
    fn brackets() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a*(b+c)");
        assert_eq!(
            dump_latex(&term, &None),
            String::from("a*\\left b+c\\right ")
        );
    }

    #[test]
    fn decoration_variable() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a");
        let path = vec![];
        let deco = Some(Decoration {
            path: &path,
            pre: "<",
            post: ">",
        });

        assert_eq!(dump_latex(&term, &deco), String::from("<a>"));
    }

    #[test]
    fn decoration_operator() {
        let context = create_context(vec![]);
        let term = Symbol::parse(&context, "a+b");
        let path = vec![0];
        let deco = Some(Decoration {
            path: &path,
            pre: "<",
            post: ">",
        });

        assert_eq!(dump_latex(&term, &deco), String::from("<a>+b"));
    }
}
