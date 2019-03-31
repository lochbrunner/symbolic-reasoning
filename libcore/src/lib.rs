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

#[macro_use]
extern crate nom;
extern crate maplit;

pub mod fit;
mod parser;
mod parsers_dep;
pub mod rule;
pub mod symbol;
pub use fit::*;
pub use rule::*;
pub use symbol::*;
