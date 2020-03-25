use std::fs::File;
extern crate serde_yaml;

#[derive(Deserialize)]
struct Files {
    #[serde(rename = "trainings-data")]
    pub trainings_data: String,
}

#[derive(Deserialize)]
struct Generation {
    pub stages: Vec<usize>,
    #[serde(rename = "max-depth")]
    pub max_depth: u32,
}

#[derive(Deserialize)]
struct Dataset {
    pub files: Files,
    pub generation: Generation,
}

pub struct Configuration {
    pub stages: Vec<usize>,
    pub dump_filename: String,
    pub max_depth: u32,
}

impl Configuration {
    pub fn load(filename: &str) -> Result<Configuration, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        let dataset: Dataset = serde_yaml::from_reader(file).map_err(|msg| msg.to_string())?;
        let Dataset { files, generation } = dataset;

        Ok(Configuration {
            stages: generation.stages,
            max_depth: generation.max_depth,
            dump_filename: files.trainings_data,
        })
    }
}
