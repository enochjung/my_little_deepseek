mod tensor;
mod tensor_slice;

pub use tensor::Tensor;
pub use tensor_slice::TensorSlice;

use super::Error;
use crate::inference::{kernel, utils};
use core::marker::PhantomData;

pub trait DataType {
    const BYTES: usize;
}

pub struct BF16;
impl DataType for BF16 {
    const BYTES: usize = 2;
}

pub struct F32;
impl DataType for F32 {
    const BYTES: usize = 4;
}

pub trait StorageType {}

pub struct Host;
impl StorageType for Host {}

#[allow(unused)]
pub struct Device;
impl StorageType for Device {}

#[derive(Clone, Copy)]
pub struct Layout<D: DataType> {
    pub offset: usize,
    pub is_row_major: bool,
    pub nrow: usize,
    pub ncol: usize,
    pub stride: usize,
    _phantom: PhantomData<D>,
}
