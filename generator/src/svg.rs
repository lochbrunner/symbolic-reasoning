use std::io::Result;
use std::io::Write;

pub trait Node {
    fn serialize_impl(&self, indent: usize, writer: &mut Write) -> Result<()>;
}

pub struct Class {
    pub name: String,
    pub color: palette::Srgb,
}

pub struct Style {
    pub classes: Vec<Class>,
}

impl Node for Style {
    fn serialize_impl(&self, indent: usize, writer: &mut Write) -> Result<()> {
        writeln!(writer, "{:indent$}<style>", "", indent = indent)?;
        for class in self.classes.iter() {
            let ident_str = format!("{:indent$}", "", indent = indent + 2);
            write!(
                writer,
                "{indent}.{name} {{\n{indent}  fill: rgb({r:.0}, {g:.0}, {b:.0});\n{indent}}}\n",
                indent = ident_str,
                name = class.name,
                r = class.color.red * 255.0,
                g = class.color.green * 255.0,
                b = class.color.blue * 255.0,
            )?;
            write!(
                writer,
                "{indent}line.{name} {{\n{indent}  stroke: rgb({r:.0}, {g:.0}, {b:.0});\n{indent}}}\n",
                indent = ident_str,
                name = class.name,
                r = class.color.red * 255.0,
                g = class.color.green * 255.0,
                b = class.color.blue * 255.0,
            )?;
        }
        writeln!(writer, "{:indent$}</style>", "", indent = indent)
    }
}

pub struct Document {
    pub childs: Vec<Box<Node>>,
    pub view_box: &'static str,
}

impl Document {
    pub fn serialize(&self, writer: &mut Write) -> Result<()> {
        writeln!(
            writer,
            "<svg viewBox=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">",
            self.view_box
        )?;
        for child in self.childs.iter() {
            child.serialize_impl(2, writer)?;
        }
        write!(writer, "</svg>")
    }
}

pub struct Text {
    pub x: f32,
    pub y: f32,
    pub content: String,
    pub class_name: String,
}

impl Node for Text {
    fn serialize_impl(&self, indent: usize, writer: &mut Write) -> Result<()> {
        write!(writer, "{:indent$}<text class=\"{class_name}\" alignment-baseline=\"middle\" font-size=\"8\" font-family=\"sans-serif\" text-anchor=\"middle\" x=\"{x:.1}\" y=\"{y:.1}\">", 
        "",indent=indent, class_name=self.class_name ,x=self.x, y=self.y)?;
        write!(writer, "{}", self.content)?;
        writeln!(writer, "</text>")
    }
}

pub struct Line {
    pub stroke_width: f32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub class_name: String,
}

impl Node for Line {
    fn serialize_impl(&self, indent: usize, writer: &mut Write) -> Result<()> {
        writeln!(writer, "{:indent$}<line class=\"{class_name}\" x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke-width=\"{stroke_width}\" />",
        "",indent=indent, class_name=self.class_name, x1=self.x1, y1=self.y1,x2=self.x2, y2=self.y2, stroke_width=self.stroke_width)
    }
}
