//! Rules are used to transform one expression into another
//!
//! # Example
//!
//! ```latex
//! A*B+A*C -> A*(B+C)
//! ```
//!
//! Can be applied to
//!
//! ```latex
//! r*s+r*t
//! ```
//!
//! which transforms it to
//!
//! ```latex
//! r*(s+t)
//! ```

#![feature(map_get_key_value)]
#![feature(test)]
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate maplit;
extern crate nom;
extern crate test;

pub mod apply;
pub mod common;
pub mod context;
pub mod dumper;
pub mod fit;
pub mod parser;
pub mod rule;
pub mod symbol;
pub use apply::*;
pub use context::*;
pub use fit::*;
pub use rule::*;
pub use symbol::*;
