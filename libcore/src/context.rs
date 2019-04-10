use std::collections::HashMap;

pub struct Declaration {
    pub is_fixed: bool,
    pub is_function: bool,
}

pub struct Context {
    pub functions: HashMap<String, Declaration>,
}

impl Context {
    pub fn is_fixed(&self, ident: &String) -> bool {
        match self.functions.get(ident) {
            None => false,
            Some(declaration) => declaration.is_fixed,
        }
    }
    pub fn is_function(&self, ident: &String) -> bool {
        match self.functions.get(ident) {
            None => false,
            Some(declaration) => declaration.is_function,
        }
    }
}
