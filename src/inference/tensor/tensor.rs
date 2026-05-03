use super::*;

use std::cell::RefCell;
use std::ops::Range;

/// Owned 2D tensor with cached pointer-backed slice views.
pub struct Tensor<D: DataType, S: StorageType> {
    data: utils::Mmap,
    layout: Layout<D>,
    slices: RefCell<Vec<Box<TensorSlice<D, S>>>>,
    _phantom: PhantomData<(D, S)>,
}

impl<D: DataType, S: StorageType> Tensor<D, S> {
    pub fn new(is_row_major: bool, nrow: usize, ncol: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn with_capacity(nrow_cap: usize, ncol: usize) -> Result<Self, Error> {
        todo!()
    }

    /// Returns a cached immutable slice view over a row and column subrange.
    pub fn as_slice<'a>(
        &'a self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<&'a TensorSlice<D, S>, Error> {
        todo!()
    }

    /// Returns a cached mutable slice view over a row and column subrange.
    pub fn as_mut_slice<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<&'a mut TensorSlice<D, S>, Error> {
        todo!()
    }

    /// Returns the tensor with transposed layout.
    pub fn transpose<'a>(&'a self) -> &'a TensorSlice<D, S> {
        todo!()
    }

    /// Returns the tensor layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `&Layout`: reference to the tensor layout.
    ///
    /// Validation:
    /// - None
    pub fn layout<'a>(&'a self) -> &'a Layout<D> {
        todo!()
    }

    /// Appends rows or columns from `other` into `self`.
    pub fn append(&mut self, other: &TensorSlice<D, S>) -> Result<(), Error> {
        todo!()
    }

    /// Returns pointer to first element of tensor data (const).
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*const u8`: pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    fn data_ptr(&self) -> *const u8 {
        todo!()
    }

    /// Returns mutable pointer to first element of tensor data.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*mut u8`: mutable pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    fn data_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }
}

impl<D: DataType, S: StorageType> Clone for Tensor<D, S> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'a> From<&TensorSlice<BF16, Host>> for Tensor<F32, Host> {
    fn from(src: &TensorSlice<BF16, Host>) -> Self {
        todo!()
    }
}

impl Tensor<F32, Host> {
    pub fn silu(&mut self) -> () {
        todo!()
    }

    pub fn rms_norm(&mut self, weight: &TensorSlice<F32, Host>, epsilon: f32) -> Result<(), Error> {
        todo!()
    }

    pub fn muladd_weight_bias(
        &mut self,
        weight: &TensorSlice<F32, Host>,
        bias: &TensorSlice<F32, Host>,
    ) -> Result<(), Error> {
        todo!()
    }
}
