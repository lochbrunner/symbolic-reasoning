use core::{Context, Symbol};
use std::fs::File;
extern crate serde_yaml;
use clap::{App, Arg, ArgMatches};

#[derive(Deserialize)]
struct Files {
    #[serde(rename = "trainings-data")]
    pub trainings_data: String,
    #[serde(rename = "trainings-data-traces")]
    pub trace_filename: String,
}

#[derive(Deserialize)]
pub struct Augmentation {
    pub enabled: bool,
    pub factor: usize,
}

#[derive(Deserialize)]
struct Generation {
    pub stages: Vec<usize>,
    #[serde(rename = "max-depth")]
    pub max_depth: u32,
    #[serde(rename = "min-working-density")]
    pub min_working_density: f32,
    #[serde(rename = "min-result-density")]
    pub min_result_density: f32,
    #[serde(rename = "blacklist-pattern")]
    pub blacklist_pattern: Vec<String>,
    #[serde(rename = "distribution-suppression-exponent")]
    pub distribution_suppression_exponent: f64,
    #[serde(rename = "max-size")]
    pub max_size: u32,
    pub augmentation: Augmentation,
}

#[derive(Deserialize)]
struct Dataset {
    pub files: Files,
    pub generation: Generation,
}

pub struct Configuration {
    pub stages: Vec<usize>,
    pub dump_filename: String,
    pub trace_filename: String,
    pub max_depth: u32,
    pub min_working_density: f32,
    pub min_result_density: f32,
    pub blacklist_pattern: Vec<Symbol>,
    pub distribution_suppression_exponent: f64,
    pub max_size: u32,
    pub augmentation: Augmentation,
}

pub trait ConfigurationOverrides {
    fn config_overrides(self) -> Self;
}

impl Configuration {
    pub fn load(filename: &str, matches: &ArgMatches) -> Result<Configuration, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        let dataset: Dataset = serde_yaml::from_reader(file).map_err(|msg| msg.to_string())?;
        let Dataset { files, generation } = dataset;

        let context = Context::standard();

        let blacklist_pattern = generation
            .blacklist_pattern
            .iter()
            .map(|s| Symbol::parse(&context, s))
            .collect::<Result<Vec<_>, _>>()?;

        let stages = if let Some(s) = matches.values_of("stages") {
            Some(s.map(|s| s.parse::<usize>().unwrap()).collect::<Vec<_>>())
        } else {
            None
        };

        Ok(Configuration {
            stages: stages.unwrap_or(generation.stages),
            max_depth: generation.max_depth,
            min_working_density: generation.min_working_density,
            min_result_density: generation.min_result_density,
            max_size: generation.max_size,
            augmentation: generation.augmentation,
            blacklist_pattern,
            distribution_suppression_exponent: generation.distribution_suppression_exponent,
            dump_filename: files.trainings_data,
            trace_filename: files.trace_filename,
        })
    }
}

impl<'a, 'b> ConfigurationOverrides for App<'a, 'b> {
    fn config_overrides(self) -> Self {
        self.arg(
            Arg::with_name("stages")
                .long("stages")
                .help("Used stages")
                .takes_value(true),
        )
    }
}
