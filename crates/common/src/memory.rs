use crate::backend::Backend;
use crate::error::Error;

pub trait MemoryRef: Send + Sync {
    type Item;
    type Base: Backend;

    fn shape(&self) -> (u32, u32);
}

pub trait MemoryMut: MemoryRef {}

pub trait Memory: MemoryMut + Sized {
    fn new(nrow: u32, ncol: u32) -> Result<Self, Error>;
    fn resize(&mut self, nrow: u32, ncol: u32) -> Result<(), Error>;
}

impl<M: Memory> MemoryRef for &M {
    type Item = M::Item;
    type Base = M::Base;

    fn shape(&self) -> (u32, u32) {
        (**self).shape()
    }
}

impl<M: Memory> MemoryRef for &mut M {
    type Item = M::Item;
    type Base = M::Base;

    fn shape(&self) -> (u32, u32) {
        (**self).shape()
    }
}

impl<M: Memory> MemoryMut for &mut M {}
