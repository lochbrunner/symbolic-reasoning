use std::collections::HashMap;
use std::fs::File;
extern crate serde_yaml;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Declaration {
    pub is_fixed: bool,
    pub is_function: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub declarations: HashMap<String, Declaration>,
}

impl Context {
    pub fn is_fixed(&self, ident: &str) -> bool {
        match self.declarations.get(ident) {
            None => false,
            Some(declaration) => declaration.is_fixed,
        }
    }
    pub fn is_function(&self, ident: &str) -> bool {
        match self.declarations.get(ident) {
            None => false,
            Some(declaration) => declaration.is_function,
        }
    }

    pub fn load(filename: &str) -> Context {
        let file = File::open(filename).expect("Opening file");
        serde_yaml::from_reader(file).expect("Deserialize context")
    }
}
