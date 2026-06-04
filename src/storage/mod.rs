mod mmap;

pub(crate) use mmap::{Mmap, MmapMut};

pub(crate) trait Location {}
pub(crate) struct Host;
impl Location for Host {}
#[allow(unused)]
pub(crate) struct Device;
impl Location for Device {}

pub(crate) trait Storage {
    type Loc: Location;

    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const ();
}

pub(crate) trait Mutable: Storage {
    fn as_mut_ptr(&mut self) -> *mut ();
    fn resize(&mut self, len: usize) -> Result<(), crate::Error>;
}

impl<'a, S: Storage> Storage for &'a S {
    type Loc = S::Loc;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn as_ptr(&self) -> *const () {
        (**self).as_ptr()
    }
}
impl<'a, M: Mutable> Storage for &'a mut M {
    type Loc = M::Loc;
    fn len(&self) -> usize {
        (**self).len()
    }
    fn as_ptr(&self) -> *const () {
        (**self).as_ptr()
    }
}
impl<'a, M: Mutable> Mutable for &'a mut M {
    fn as_mut_ptr(&mut self) -> *mut () {
        (**self).as_mut_ptr()
    }
    fn resize(&mut self, len: usize) -> Result<(), crate::Error> {
        (**self).resize(len)
    }
}

pub(crate) trait Owned: Storage {
    type ReadOnly: Storage<Loc = Self::Loc> + Owned;

    fn into_readonly(self) -> Self::ReadOnly;
}
