mod mmap;

pub(crate) use mmap::{Mmap, MmapMut};

pub(crate) trait Storage {
    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const ();
}
impl Storage for Mmap {
    fn len(&self) -> usize {
        self.len()
    }
    fn as_ptr(&self) -> *const () {
        self.as_ptr()
    }
}
impl Storage for MmapMut {
    fn len(&self) -> usize {
        self.len()
    }
    fn as_ptr(&self) -> *const () {
        self.as_ptr()
    }
}
impl<'a> Storage for &'a Mmap {
    fn len(&self) -> usize {
        (*self).len()
    }
    fn as_ptr(&self) -> *const () {
        (*self).as_ptr()
    }
}
