#![feature(stdarch_x86_avx512_bf16)]

mod backend;
mod elem_type;
mod error;
mod matrix_layout;
mod memory;

pub use backend::{Backend, BackendOps};
pub use elem_type::ElemType;
pub use error::MLTError;
pub use matrix_layout::MatrixLayout;
pub use memory::{Memory, MemoryMut, MemoryOwn};
