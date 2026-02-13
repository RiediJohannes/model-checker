pub mod solving;
pub mod resolution;

mod types;

pub use types::*;
pub use solving::Literal;
pub use solving::{VAR_OFFSET,TRUE,FALSE};
