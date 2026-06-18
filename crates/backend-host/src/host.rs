use core::Backend;

use crate::mmap::Mmap;

use std::marker::PhantomData;

pub struct Host<T> {
    _phantom: PhantomData<T>,
}

impl<T> Backend<T> for Host<T> {
    type Operand = Mmap<T>;
}
