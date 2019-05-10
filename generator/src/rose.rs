use crate::svg;
use crate::trace::Trace;
use palette;
use std::collections::HashMap;
use std::f32;
use std::fs::File;
use std::io;

/// Paints a rose from the information of the trace
pub fn draw_rose(path: &str, trace: &Trace) -> io::Result<()> {
    let height = 256;
    let width = 256;

    let initial = svg::Text {
        x: (width / 2) as f32,
        y: (height / 2) as f32,
        content: trace.initial.to_string(),
        class_name: "black".to_string(),
    };


    let mut childs: Vec<Box<svg::Node>> = Vec::new();

    // Create a class for each rule
    let mut rules_styles = HashMap::new();
    let mut rules_i = 0;
    for deduced in trace.stage.iter(){
        let key = deduced.rule.to_string();
        let name = format!("rule_{}", rules_i);
        if !rules_styles.contains_key(&key) {
            rules_styles.insert(deduced.rule.to_string(), name);
            rules_i += 1;
        }
    }
    let total_rules = rules_styles.len();
        let mut style = svg::Style {
            classes: vec![]};

    for i in 0..total_rules {
        let angle = 360.0 * (i as f32) / (total_rules as f32);
        let name = format!("rule_{}", i);
            let c = svg::Class {
                name,
                color: palette::Srgb::from(palette::Hsv::new(
                    palette::RgbHue::from_degrees(angle),
                    1.0,
                    0.6,
                ),
            
        };
        style.classes.push(c);
    }
    childs.push(Box::new(style));

    let total_deduced = trace.stage.len();
    for (i, deduced) in trace.stage.iter().enumerate() {
        let r = 64.0;
        let mx = (width as f32) / 2.0;
        let my = (width as f32) / 2.0;
        let angle = 2.0 * f32::consts::PI * (i as f32) / (total_deduced as f32);
        let x = r * angle.cos() + mx;
        let y = r * angle.sin() + my;
        let class_name = rules_styles.get(&deduced.rule.to_string()).expect("Rule");
        let text = svg::Text {
            x,
            y,
            class_name: class_name.clone(),
            content: deduced.deduced.to_string(),
        };
        childs.push(Box::new(text));

        let margin = 10.0;
        let outer_r = r - margin;
        let x1 = margin * angle.cos() + mx;
        let y1 = margin * angle.sin() + mx;
        let x2 = outer_r * angle.cos() + mx;
        let y2 = outer_r * angle.sin() + mx;
        let line = svg::Line {
            stroke_width: 1.0,
            x1,
            y1,
            x2,
            y2,
            class_name: class_name.clone(),
        };
        childs.push(Box::new(line));
    }

    childs.push(Box::new(initial));


    let document = svg::Document {
        view_box: "0 0 256 256",
        childs,
    };

    let mut file = File::create(path)?;

    document.serialize(&mut file)
}
