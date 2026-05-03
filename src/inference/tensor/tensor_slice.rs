use super::*;

use std::ops::Range;

/// Pointer-backed tensor slice managed by a parent `Tensor`.
pub struct TensorSlice<D: DataType, S: StorageType> {
    data: *mut u8,
    offset_bytes: usize,
    layout: Layout<D>,
    _phantom: PhantomData<(D, S)>,
}

impl<D: DataType, S: StorageType> TensorSlice<D, S> {
    /// Constructs a pointer-backed tensor slice.
    fn new(
        data: *mut u8,
        data_bytes: usize,
        offset_bytes: usize,
        is_row_major: bool,
        nrow: usize,
        ncol: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        todo!()
    }

    /// Returns an immutable subview of this `TensorSlice`.
    pub fn as_slice(
        &self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<&TensorSlice<D, S>, Error> {
        todo!()
    }

    /// Returns a mutable subview of this `TensorSlice`.
    pub fn as_mut_slice(
        &mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<&mut TensorSlice<D, S>, Error> {
        todo!()
    }

    /// Returns the tensor with transposed layout.
    pub fn transpose(&self) -> TensorSlice<D, S> {
        todo!()
    }

    /// Returns the tensor layout.
    pub fn layout<'a>(&'a self) -> &'a Layout<D> {
        todo!()
    }

    /// Returns pointer to first element of tensor data (const).
    pub fn data_ptr(&self) -> *const u8 {
        todo!()
    }

    /// Returns mutable pointer to first element of tensor data.
    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        todo!()
    }

    /// Splits the tensor by rows into two writable views.
    pub fn split_row(
        &mut self,
        mid: usize,
    ) -> Result<(&mut TensorSlice<D, S>, &mut TensorSlice<D, S>), Error> {
        todo!()
    }

    /// Splits the tensor by columns into two writable views.
    pub fn split_col(
        &mut self,
        mid: usize,
    ) -> Result<(&mut TensorSlice<D, S>, &mut TensorSlice<D, S>), Error> {
        todo!()
    }
}
