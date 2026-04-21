mod blas;
mod operable;

use super::Error;
use std::marker::PhantomData;
use std::ops::Range;

pub use operable::{BinaryOperable, UnaryOperable};

pub trait DataType {
    const BYTES: usize;
}

pub struct BF16;

pub struct F32;

impl DataType for BF16 {
    const BYTES: usize = 2;
}

impl DataType for F32 {
    const BYTES: usize = 4;
}

pub trait StorageType {}

pub struct HostMemory {
    data: Vec<u8>,
}

pub struct HostMemoryRef<'a> {
    data: &'a [u8],
}

#[allow(unused)]
pub struct DeviceMemory;

impl StorageType for HostMemory {}
impl<'a> StorageType for HostMemoryRef<'a> {}
impl StorageType for DeviceMemory {}

impl Clone for HostMemory {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<'a> Clone for HostMemoryRef<'a> {
    fn clone(&self) -> Self {
        Self { data: self.data }
    }
}

pub struct Tensor<D: DataType, S: StorageType> {
    storage: S,
    nrow: usize,
    ncol: usize,
    stride: usize,
    is_trans: bool,
    _phantom: PhantomData<D>,
}

impl<D: DataType, S: StorageType + Clone> Clone for Tensor<D, S> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            nrow: self.nrow,
            ncol: self.ncol,
            stride: self.stride,
            is_trans: self.is_trans,
            _phantom: PhantomData,
        }
    }
}

impl<D: DataType, S: StorageType> Tensor<D, S> {
    pub fn shape(&self) -> [usize; 2] {
        if !self.is_trans {
            [self.nrow, self.ncol]
        } else {
            [self.ncol, self.nrow]
        }
    }

    pub fn is_row_contiguous(&self) -> bool {
        !self.is_trans
    }
}

trait HostBytes {
    fn bytes(&self) -> &[u8];
}

impl HostBytes for HostMemory {
    fn bytes(&self) -> &[u8] {
        &self.data
    }
}

impl<'a> HostBytes for HostMemoryRef<'a> {
    fn bytes(&self) -> &[u8] {
        self.data
    }
}

fn slice_host_memory<'a, D: DataType>(
    data: &'a [u8],
    stride: usize,
    is_trans: bool,
    slice_rows: Range<usize>,
    slice_cols: Range<usize>,
) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
    let (slice_rows, slice_cols) = if is_trans {
        (slice_cols, slice_rows)
    } else {
        (slice_rows, slice_cols)
    };

    let slice_nrow = slice_rows.end - slice_rows.start;
    let slice_ncol = slice_cols.end - slice_cols.start;

    let start = (slice_rows.start * stride + slice_cols.start) * D::BYTES;
    let end = (slice_rows.end * stride + slice_cols.end) * D::BYTES;

    let view = data
        .get(start..end)
        .ok_or_else(|| Error::out_of_bound(end, data.len()))?;

    Ok(Tensor {
        storage: HostMemoryRef { data: view },
        nrow: slice_nrow,
        ncol: slice_ncol,
        stride,
        is_trans,
        _phantom: PhantomData,
    })
}

impl<'a, D: DataType> Tensor<D, HostMemoryRef<'a>> {
    pub fn new(
        data: &'a [u8],
        nrow: usize,
        ncol: usize,
        is_row_major: bool,
    ) -> Result<Self, Error> {
        let volume = D::BYTES
            .checked_mul(nrow)
            .and_then(|v| v.checked_mul(ncol))
            .ok_or_else(|| Error::shape_mismatch(0, data.len()))?;
        if volume != data.len() {
            return Err(Error::shape_mismatch(volume, data.len()));
        }

        let (nrow, ncol, stride, is_trans) = if is_row_major {
            (nrow, ncol, ncol, false)
        } else {
            (ncol, nrow, nrow, true)
        };

        Ok(Self {
            storage: HostMemoryRef { data },
            nrow,
            ncol,
            stride,
            is_trans,
            _phantom: PhantomData,
        })
    }

    pub fn slice(
        &self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        slice_host_memory(self.storage.data, self.stride, self.is_trans, rows, cols)
    }
}

impl<D: DataType> Tensor<D, HostMemory> {
    pub fn new(data: Vec<u8>, nrow: usize, ncol: usize, is_row_major: bool) -> Result<Self, Error> {
        let volume = D::BYTES
            .checked_mul(nrow)
            .and_then(|v| v.checked_mul(ncol))
            .ok_or_else(|| Error::shape_mismatch(0, data.len()))?;
        if volume != data.len() {
            return Err(Error::shape_mismatch(volume, data.len()));
        }

        let (nrow, ncol, stride, is_trans) = if is_row_major {
            (nrow, ncol, ncol, false)
        } else {
            (ncol, nrow, nrow, true)
        };

        Ok(Self {
            storage: HostMemory { data },
            nrow,
            ncol,
            stride,
            is_trans,
            _phantom: PhantomData,
        })
    }

    pub fn with_capacity(capacity: usize, ncol: usize) -> Result<Self, Error> {
        let bytes_capacity = capacity
            .checked_mul(D::BYTES)
            .ok_or_else(|| Error::shape_mismatch(0, capacity))?;
        let data = Vec::with_capacity(bytes_capacity);

        Ok(Self {
            storage: HostMemory { data },
            nrow: 0,
            ncol,
            stride: ncol,
            is_trans: false,
            _phantom: PhantomData,
        })
    }

    pub fn slice<'a>(
        &'a self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        slice_host_memory(&self.storage.data, self.stride, self.is_trans, rows, cols)
    }

    /// Appends `other` to `self` by aligning logical columns, even when transposed states differ.
    ///
    /// Both tensors must have the same logical column count.
    /// The result keeps `self`'s transpose state and only increases row count.
    ///
    /// Example:
    /// - `self`: `{ nrow: sr, ncol: 4, is_trans: true }`
    /// - `other`: `{ nrow: or, ncol: 4, is_trans: false }`
    /// - result: `{ nrow: sr + or, ncol: 4, is_trans: true }`
    pub fn append<'a>(&mut self, _other: &Tensor<D, HostMemoryRef<'a>>) -> Result<(), Error> {
        todo!()
    }
}

impl<'a> From<&Tensor<BF16, HostMemoryRef<'a>>> for Tensor<F32, HostMemory> {
    fn from(_value: &Tensor<BF16, HostMemoryRef<'a>>) -> Self {
        todo!()
    }
}
