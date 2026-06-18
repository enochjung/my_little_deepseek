#![feature(stdarch_x86_avx512_bf16)]
#![feature(min_specialization)]

mod host;
mod host_bf16;
mod host_f32;
mod kernel;
mod mmap;

pub use host::Host;
pub use mmap::Mmap;
