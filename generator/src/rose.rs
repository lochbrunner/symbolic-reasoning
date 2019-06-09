use crate::iter_extensions::{PickTraitVec, Strategy};
use crate::svg;
use core::{Rule, Trace};
use palette;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::io;

struct RuleStyleManager {
    i: i32,
    map: HashMap<String, String>,
}

impl<'a> RuleStyleManager {
    pub fn new() -> RuleStyleManager {
        RuleStyleManager {
            map: HashMap::new(),
            i: 0,
        }
    }
    pub fn get(&mut self, rule: &Rule) -> String {
        let key = rule.to_string();
        if !self.map.contains_key(&key) {
            let name = format!("rule-{}", self.i);
            self.map.insert(key.clone(), name);
            self.i += 1;
        }
        self.map.get(&key).expect("").clone()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn classes(&self) -> Vec<svg::Class> {
        let total_rules = self.len();
        self.map
            .iter()
            .enumerate()
            .map(|(i, (_, name))| {
                let angle = 360.0 * (i as f32) / (total_rules as f32);
                svg::Class {
                    name: name.clone(),
                    color: palette::Srgb::from(palette::Hsv::new(
                        palette::RgbHue::from_degrees(angle),
                        1.0,
                        0.6,
                    )),
                }
            })
            .collect()
    }
}

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

    let mut childs: Vec<Box<dyn svg::Node>> = Vec::new();
    let mut rules_styles = RuleStyleManager::new();

    let num_stage_1 = 7;
    let num_stage_2 = 4;
    for (i, stage1) in trace
        .stages
        .pick(Strategy::Uniform(num_stage_1))
        .enumerate()
    {
        let r1 = 56.0;
        let mx = (width as f32) / 2.0;
        let my = (height as f32) / 2.0;
        let angle = 2.0 * PI * (i as f32) / (num_stage_1 as f32);
        let x = r1 * angle.cos() + mx;
        let y = r1 * angle.sin() + my;
        let class_name = rules_styles.get(stage1.info.rule);
        let text = svg::Text {
            x,
            y,
            class_name: class_name.clone(),
            content: stage1.info.deduced.to_string(),
        };
        childs.push(Box::new(text));

        let margin = 12.0;
        let outer_r = r1 - margin;
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

        // Second stage
        let r2 = 112.0;
        let max_spread_angle = 0.7 * PI / (num_stage_1 as f32);
        let spread_delta = 2.0 * max_spread_angle / (num_stage_2 as f32 - 1.0);
        for (j, stage2) in stage1
            .successors
            .pick(Strategy::Uniform(num_stage_2))
            .enumerate()
        {
            let inner_angle = angle - max_spread_angle + (j as f32) * spread_delta;
            let x1 = (r2 - margin) * inner_angle.cos() + mx;
            let y1 = (r2 - margin) * inner_angle.sin() + mx;
            let x2_m = (r1) * angle.cos() + mx;
            let y2_m = (r1) * angle.sin() + mx;
            let x2 = x2_m * 0.8 + x1 * 0.2;
            let y2 = y2_m * 0.8 + y1 * 0.2;
            let class_name = rules_styles.get(stage2.info.rule);

            let line = svg::Line {
                stroke_width: 1.0,
                x1,
                y1,
                x2,
                y2,
                class_name: class_name.clone(),
            };
            childs.push(Box::new(line));

            let x = (r2 - 0.0) * inner_angle.cos() + mx;
            let y = (r2 - 0.0) * inner_angle.sin() + mx;

            let text = svg::Text {
                x,
                y,
                class_name: class_name.clone(),
                content: stage2.info.deduced.to_string(),
            };
            childs.push(Box::new(text));
        }
    }

    childs.push(Box::new(initial));

    let style = svg::Style {
        classes: rules_styles.classes(),
    };
    childs.push(Box::new(style));

    let document = svg::Document {
        view_box: [0, 0, width, height],
        childs,
    };

    let mut file = File::create(path)?;

    document.serialize(&mut file)
}
