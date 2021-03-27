use crate::symbol::Symbol;
use std::collections::HashMap;
use std::fs::File;
extern crate serde_yaml;

fn default_as_true() -> bool {
    true
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Declaration {
    #[serde(default = "default_as_true")]
    pub is_fixed: bool,
    #[serde(default)]
    pub is_function: bool,
    #[serde(default)]
    pub only_root: bool,
}

impl Declaration {
    pub fn function(is_fixed: bool) -> Self {
        Self {
            is_fixed,
            is_function: true,
            only_root: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub declarations: HashMap<String, Declaration>,
}

impl Context {
    pub fn flags(&self, ident: &str) -> u32 {
        match self.declarations.get(ident) {
            None => 0,
            Some(declaration) => Symbol::create_flags(declaration.is_fixed, declaration.only_root),
        }
    }

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
    pub fn only_root(&self, ident: &str) -> bool {
        match self.declarations.get(ident) {
            None => false,
            Some(declaration) => declaration.only_root,
        }
    }

    pub fn load(filename: &str) -> Result<Context, String> {
        let file = File::open(filename).map_err(|msg| msg.to_string())?;
        serde_yaml::from_reader(file).map_err(|msg| msg.to_string())
    }

    pub fn register_standard_operators(&mut self) {
        fn dec(only_root: bool) -> Declaration {
            Declaration {
                is_fixed: true,
                is_function: true,
                only_root,
            }
        }
        self.declarations.insert("=".to_string(), dec(true));
        self.declarations.insert("+".to_string(), dec(false));
        self.declarations.insert("-".to_string(), dec(false));
        self.declarations.insert("*".to_string(), dec(false));
        self.declarations.insert("/".to_string(), dec(false));
        self.declarations.insert("^".to_string(), dec(false));
        self.declarations.insert("!".to_string(), dec(false));
        self.declarations.insert(">".to_string(), dec(true));
        self.declarations.insert("<".to_string(), dec(true));
        self.declarations.insert("!=".to_string(), dec(true));
        self.declarations.insert(">=".to_string(), dec(true));
        self.declarations.insert("<=".to_string(), dec(true));
    }

    /// Creates a context with the standard operations registered
    pub fn standard() -> Context {
        let mut context = Context {
            declarations: HashMap::new(),
        };
        context.register_standard_operators();
        context
    }
}
