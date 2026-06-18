use crate::{Backend, ElemType, MLTError};
use std::convert::TryFrom;

pub trait Memory<T: ElemType>: Send + Sync {
    type Base: MemoryOwn<T>;

    fn as_base(&self) -> &Self::Base;
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const T;
}

pub trait MemoryMut<T: ElemType>: Memory<T> {
    fn as_mut_base(&mut self) -> &mut Self::Base;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait MemoryOwn<T: ElemType>:
    MemoryMut<T, Base = Self> + TryFrom<std::fs::File, Error = MLTError>
{
    type Backend: Backend<T, Memory = Self>;

    fn new(size: usize) -> Result<Self, MLTError>;
    fn resize(&mut self, size: usize) -> Result<(), MLTError>;
}

impl<T: ElemType, MO: MemoryOwn<T>> Memory<T> for &MO {
    type Base = MO;

    fn size(&self) -> usize {
        (**self).size()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const T {
        (**self).as_ptr()
    }
}

impl<T: ElemType, MO: MemoryOwn<T>> Memory<T> for &mut MO {
    type Base = MO;

    fn size(&self) -> usize {
        (**self).size()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const T {
        (**self).as_ptr()
    }
}

impl<T: ElemType, MO: MemoryOwn<T>> MemoryMut<T> for &mut MO {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        (**self).as_mut_base()
    }
    fn as_mut_ptr(&mut self) -> *mut T {
        (**self).as_mut_ptr()
    }
}
