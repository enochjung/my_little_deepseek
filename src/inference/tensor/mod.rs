mod operable;

use super::Error;
use std::marker::PhantomData;

//pub use operable::{BinaryOperable, UnaryOperable};

const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;

pub trait DataType {}
pub trait StorageType {}

//#[derive(Debug, Clone, Copy)]
pub struct BF16;

//#[derive(Debug, Clone, Copy)]
pub struct F32;

impl DataType for BF16 {}
impl DataType for F32 {}

pub struct HostMemory {
    #[allow(unused)]
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

// Tensor::storage matrix format is always row-major.
pub struct Tensor<D: DataType, S: StorageType> {
    storage: S,
    nrow: usize,
    ncol: usize,
    transposed: bool,
    _phantom: PhantomData<D>,
}

impl<D: DataType, S: StorageType> Tensor<D, S> {
    #[allow(unused)]
    pub fn shape(&self) -> [usize; 2] {
        if !self.transposed {
            [self.nrow, self.ncol]
        } else {
            [self.ncol, self.nrow]
        }
    }

    #[allow(unused)]
    pub fn is_row_contiguous(&self) -> bool {
        !self.transposed
    }
}

impl<'a, D: DataType> Tensor<D, HostMemoryRef<'a>> {
    #[allow(unused)]
    pub fn get_row(&self, _index: usize) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        todo!()
    }
}

impl<D: DataType> Tensor<D, HostMemory> {
    pub fn get_row<'a>(&'a self, _index: usize) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        todo!()
    }
}

impl<'a> Tensor<BF16, HostMemoryRef<'a>> {
    pub fn new(data: &'a [u8], shape: [usize; 2], is_row_major: bool) -> Result<Self, Error> {
        let volume = BF16_BYTES * shape[0] * shape[1];
        if volume != data.len() {
            return Err(Error::shape_mismatch(volume, data.len()));
        }

        let (nrow, ncol, transposed) = if is_row_major {
            (shape[0], shape[1], false)
        } else {
            (shape[1], shape[0], true)
        };

        Ok(Self {
            storage: HostMemoryRef { data },
            nrow,
            ncol,
            transposed,
            _phantom: PhantomData,
        })
    }
}

impl Tensor<F32, HostMemory> {
    #[allow(unused)]
    pub fn new(data: Vec<u8>, shape: [usize; 2], is_row_major: bool) -> Result<Self, Error> {
        let volume = F32_BYTES * shape[0] * shape[1];
        if volume != data.len() {
            return Err(Error::shape_mismatch(volume, data.len()));
        }

        let (nrow, ncol, transposed) = if is_row_major {
            (shape[0], shape[1], false)
        } else {
            (shape[1], shape[0], true)
        };

        Ok(Self {
            storage: HostMemory { data },
            nrow,
            ncol,
            transposed,
            _phantom: PhantomData,
        })
    }
}

impl<'a> From<&Tensor<BF16, HostMemoryRef<'a>>> for Tensor<F32, HostMemory> {
    fn from(value: &Tensor<BF16, HostMemoryRef<'a>>) -> Self {
        let bytes = value.storage.data;
        let n = bytes.len() / BF16_BYTES;

        let mut data = vec![0; n * F32_BYTES];

        for i in 0..n {
            let hi = bytes[i * BF16_BYTES];
            let lo = bytes[i * BF16_BYTES + 1];
            data[i * F32_BYTES] = hi;
            data[i * F32_BYTES + 1] = lo;
        }

        Tensor {
            storage: HostMemory { data },
            nrow: value.nrow,
            ncol: value.ncol,
            transposed: value.transposed,
            _phantom: PhantomData,
        }
    }
}
