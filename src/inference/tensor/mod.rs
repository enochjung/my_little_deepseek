use super::Error;
use crate::inference::{kernel, utils::Mmap};
use core::marker::PhantomData;
use std::ops::Range;

pub trait Numeric {
    const BYTES: usize;
}

pub struct BF16;
impl Numeric for BF16 {
    const BYTES: usize = 2;
}

pub struct F32;
impl Numeric for F32 {
    const BYTES: usize = 4;
}

pub trait Storage {}

impl Storage for Mmap {}

pub struct Tensor<N: Numeric, S: Storage> {
    storage: S,
    offset: usize,
    is_row_major: bool,
    nrow: u32,
    ncol: u32,
    stride: usize,
    _phantom: PhantomData<N>,
}

pub struct TensorRef<'a, N: Numeric, S: Storage> {
    storage: &'a S,
    offset: usize,
    is_row_major: bool,
    nrow: u32,
    ncol: u32,
    stride: usize,
    _phantom: PhantomData<N>,
}

pub struct TensorMut<'a, N: Numeric, S: Storage> {
    storage: &'a mut S,
    offset: usize,
    is_row_major: bool,
    nrow: u32,
    ncol: u32,
    stride: usize,
    _phantom: PhantomData<N>,
}

pub trait ReadableTensor<N: Numeric, S: Storage> {
    fn as_ref(&self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorRef<'_, N, S>, Error>;
    fn as_ptr(&self) -> *const u8;
    fn transpose(self) -> Self;
}

impl<N: Numeric, S: Storage> ReadableTensor<N, S> for Tensor<N, S> {
    fn as_ref(&self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorRef<'_, N, S>, Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(self) -> Self {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> ReadableTensor<N, S> for TensorRef<'a, N, S> {
    fn as_ref(&self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorRef<'_, N, S>, Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(self) -> Self {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> ReadableTensor<N, S> for TensorMut<'a, N, S> {
    fn as_ref(&self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorRef<'_, N, S>, Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(self) -> Self {
        todo!()
    }
}

pub trait MutableTensor<N: Numeric, S: Storage> {
    fn as_mut(&mut self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorMut<'_, N, S>, Error>;
    fn as_mut_ptr(&mut self) -> *mut u8;
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for Tensor<N, S> {
    fn as_mut(&mut self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorMut<'_, N, S>, Error> {
        todo!()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for TensorMut<'a, N, S> {
    fn as_mut(&mut self, rows: Range<u32>, cols: Range<u32>) -> Result<TensorMut<'_, N, S>, Error> {
        todo!()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }
}

impl<N: Numeric, S: Storage> Tensor<N, S> {
    pub fn new(storage: S, is_row_major: bool, nrow: u32, ncol: u32) -> Result<Self, Error> {
        todo!()
    }

    pub fn append<R: ReadableTensor<N, S>>(&mut self, other: &R) -> Result<(), Error> {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> TensorRef<'a, N, S> {
    pub fn new(
        storage: &'a S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: usize,
    ) -> Result<Self, Error> {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> TensorMut<'a, N, S> {
    pub fn new(
        storage: &'a S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: usize,
    ) -> Result<Self, Error> {
        todo!()
    }

    pub fn split_row(self, mid: usize) -> Result<(Self, Self), Error> {
        todo!()
    }

    pub fn split_col(self, mid: usize) -> Result<(Self, Self), Error> {
        todo!()
    }
}

impl<N: Numeric, S: Storage> Clone for Tensor<N, S> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'a> From<&TensorRef<'a, BF16, Mmap>> for Tensor<F32, Mmap> {
    fn from(src: &TensorRef<'a, BF16, Mmap>) -> Self {
        todo!()
    }
}

impl Tensor<F32, Mmap> {
    pub fn silu(&mut self) -> () {
        todo!()
    }

    pub fn rms_norm(
        &mut self,
        weight: &TensorRef<'_, F32, Mmap>,
        epsilon: f32,
    ) -> Result<(), Error> {
        todo!()
    }

    pub fn muladd_weight_bias(
        &mut self,
        weight: &TensorRef<'_, F32, Mmap>,
        bias: &TensorRef<'_, F32, Mmap>,
    ) -> Result<(), Error> {
        todo!()
    }
}
