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
#![feature(test)]
#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate maplit;
#[cfg(test)]
extern crate p_macro;
#[cfg(test)]
extern crate stringreader;
#[macro_use]
#[cfg(test)]
extern crate vector_assertions;
extern crate nom;
extern crate test;

pub mod apply;
pub mod common;
pub mod context;
pub mod dumper;
pub mod fit;
pub mod io;
pub mod parser;
pub mod rule;
pub mod symbol;
pub use apply::*;
pub use context::*;
pub use fit::*;
pub use rule::*;
pub use symbol::*;
pub mod scenario;
pub mod solver;
