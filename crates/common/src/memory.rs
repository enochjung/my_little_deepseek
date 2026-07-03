use crate::error::Error;

pub trait Memory: Send + Sync {
    type Item;
    type Base: MemoryOwn;

    fn as_base(&self) -> &Self::Base;
    fn size(&self) -> usize;
    fn as_ptr(&self) -> *const Self::Item;
}

pub trait MemoryMut: Memory {
    fn as_mut_base(&mut self) -> &mut Self::Base;
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
}

pub trait MemoryOwn: MemoryMut<Base = Self> + Sized {
    fn new(size: usize) -> Result<Self, Error>;
    fn resize(&mut self, size: usize) -> Result<(), Error>;
}

impl<MO: MemoryOwn> Memory for &MO {
    type Item = MO::Item;
    type Base = MO;

    fn size(&self) -> usize {
        (**self).size()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const Self::Item {
        (**self).as_ptr()
    }
}

impl<MO: MemoryOwn> Memory for &mut MO {
    type Item = MO::Item;
    type Base = MO;

    fn size(&self) -> usize {
        (**self).size()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const Self::Item {
        (**self).as_ptr()
    }
}

impl<MO: MemoryOwn> MemoryMut for &mut MO {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        (**self).as_mut_base()
    }
    fn as_mut_ptr(&mut self) -> *mut Self::Item {
        (**self).as_mut_ptr()
    }
}
