mod numeric;

use crate::storage::*;
use core::marker::PhantomData;
pub(crate) use numeric::*;
use std::cell::RefCell;
use std::ops::Range;

pub(crate) struct Layout<N: Numeric> {
    offset: usize,
    nrow: u32,
    ncol: u32,
    stride: u32,
    is_row_major: bool,
    _phantom: PhantomData<N>,
}

pub(crate) struct TensorOwn<N: Numeric, S: Storage> {
    storage: S,
    layout: RefCell<Layout<N>>,
}

pub(crate) struct TensorRef<'a, N: Numeric, S: Storage> {
    storage: &'a S,
    layout: RefCell<Layout<N>>,
}

pub(crate) struct TensorMut<'a, N: Numeric, S: Storage> {
    storage: &'a mut S,
    layout: RefCell<Layout<N>>,
}

pub(crate) trait Tensor<N: Numeric, S: Storage> {
    fn as_ref(
        &self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorRef<'_, N, S>, crate::Error>;
    fn as_ptr(&self) -> *const u8;
    fn transpose(&self) -> &Self;
}

pub(crate) trait MutableTensor<N: Numeric, S: Storage>: Tensor<N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error>;
    fn as_mut_ptr(&mut self) -> *mut u8;
}

impl<N: Numeric, S: Storage> Tensor<N, S> for TensorOwn<N, S> {
    fn as_ref(
        &self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorRef<'_, N, S>, crate::Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(&self) -> &Self {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> Tensor<N, S> for TensorRef<'a, N, S> {
    fn as_ref(
        &self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorRef<'_, N, S>, crate::Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(&self) -> &Self {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> Tensor<N, S> for TensorMut<'a, N, S> {
    fn as_ref(
        &self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorRef<'_, N, S>, crate::Error> {
        todo!()
    }

    fn as_ptr(&self) -> *const u8 {
        todo!()
    }

    fn transpose(&self) -> &Self {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for TensorOwn<N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error> {
        todo!()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for TensorMut<'a, N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error> {
        todo!()
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }
}

impl<N: Numeric, S: Storage> TensorOwn<N, S> {
    pub(crate) fn new(
        storage: S,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
    ) -> Result<Self, crate::Error> {
        todo!()
    }

    pub(crate) fn append<T: Tensor<N, S>>(&mut self, other: &T) -> Result<(), crate::Error> {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> TensorRef<'a, N, S> {
    pub(crate) fn new(
        storage: &'a S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<Self, crate::Error> {
        todo!()
    }
}

impl<'a, N: Numeric, S: Storage> TensorMut<'a, N, S> {
    pub(crate) fn new(
        storage: &'a S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<Self, crate::Error> {
        todo!()
    }

    pub(crate) fn split_row(self, mid: usize) -> Result<(Self, Self), crate::Error> {
        todo!()
    }

    pub(crate) fn split_col(self, mid: usize) -> Result<(Self, Self), crate::Error> {
        todo!()
    }
}

impl<N: Numeric, S: Storage> Clone for TensorOwn<N, S> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'a, T: Tensor<BF16, S>, S: Storage> From<&T> for TensorOwn<F32, S> {
    fn from(src: &T) -> Self {
        todo!()
    }
}

impl TensorOwn<F32, Host> {
    pub(crate) fn silu(&mut self) -> () {
        todo!()
    }

    pub(crate) fn rms_norm(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        epsilon: f32,
    ) -> Result<(), crate::Error> {
        todo!()
    }

    pub(crate) fn muladd_weight_bias(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        bias: &TensorRef<'_, F32, Host>,
    ) -> Result<(), crate::Error> {
        todo!()
    }
}
