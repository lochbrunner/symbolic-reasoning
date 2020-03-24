use std::fs::File;
extern crate serde_yaml;

#[derive(Serialize, Deserialize)]
pub struct Configuration {
    pub scenario: String,
    pub stages: Vec<usize>,
    #[serde(rename = "dump-filename")]
    pub dump_filename: String,
}

impl Configuration {
    pub fn load(filename: &str) -> Result<Configuration, String> {
        let file = match File::open(filename) {
            Ok(f) => f,
            Err(msg) => return Err(msg.to_string()),
        };
        match serde_yaml::from_reader(file) {
            Ok(r) => Ok(r),
            Err(msg) => Err(msg.to_string()),
        }
    }
}
