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

mod parsers;
pub mod rule;
pub mod symbol;
pub use rule::*;
pub use symbol::*;
