#![no_std]

mod backend;
mod elem_type;
mod error;
mod matrix_layout;
mod memory;

pub use backend::Backend;
pub use elem_type::{BF16, ElemType};
pub use error::Error;
pub use matrix_layout::MatrixLayout;
pub use memory::{Memory, MemoryMut, MemoryOwn};
