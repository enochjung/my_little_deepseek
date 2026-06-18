use crate::backend::Backend;
use crate::error::MLTError;

pub trait Memory<T>: Send + Sync {
    type Base: MemoryOwn<T>;

    fn as_base(&self) -> &Self::Base;
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const T;
}

pub trait MemoryMut<T>: Memory<T> {
    fn as_mut_base(&mut self) -> &mut Self::Base;
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait MemoryOwn<T>: MemoryMut<T, Base = Self> + Sized {
    type Operator: Backend<T, Operand = Self>;

    fn new(size: usize) -> Result<Self, MLTError>;
    fn resize(&mut self, size: usize) -> Result<(), MLTError>;
}

impl<T, MO: MemoryOwn<T>> Memory<T> for &MO {
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

impl<T, MO: MemoryOwn<T>> Memory<T> for &mut MO {
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

impl<T, MO: MemoryOwn<T>> MemoryMut<T> for &mut MO {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        (**self).as_mut_base()
    }
    fn as_mut_ptr(&mut self) -> *mut T {
        (**self).as_mut_ptr()
    }
}
