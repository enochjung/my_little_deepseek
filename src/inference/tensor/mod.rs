mod operable;

use super::Error;
use std::marker::PhantomData;

//pub use operable::{BinaryOperable, UnaryOperable};

pub trait DataType {
    const BYTES: usize;
}
pub trait StorageType {}

//#[derive(Debug, Clone, Copy)]
pub struct BF16;

//#[derive(Debug, Clone, Copy)]
pub struct F32;

impl DataType for BF16 {
    const BYTES: usize = 2;
}

impl DataType for F32 {
    const BYTES: usize = 4;
}

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
    pub fn get_row(&self, index: usize) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        if index >= self.nrow {
            return Err(Error::out_of_bound(index, self.nrow));
        }

        let row_bytes = self.ncol * D::BYTES;
        let start = index * row_bytes;
        let end = start + row_bytes;
        let row_data = &self.storage.data[start..end];

        Ok(Tensor {
            storage: HostMemoryRef { data: row_data },
            nrow: 1,
            ncol: self.ncol,
            transposed: false,
            _phantom: PhantomData,
        })
    }
}

impl<D: DataType> Tensor<D, HostMemory> {
    #[allow(unused)]
    pub fn get_row<'a>(&'a self, index: usize) -> Result<Tensor<D, HostMemoryRef<'a>>, Error> {
        if index >= self.nrow {
            return Err(Error::out_of_bound(index, self.nrow));
        }

        let row_bytes = self.ncol * D::BYTES;
        let start = index * row_bytes;
        let end = start + row_bytes;
        let row_data = &self.storage.data[start..end];

        Ok(Tensor {
            storage: HostMemoryRef { data: row_data },
            nrow: 1,
            ncol: self.ncol,
            transposed: false,
            _phantom: PhantomData,
        })
    }

    #[allow(unused)]
    pub fn append(&mut self, other: &Tensor<D, HostMemoryRef>) -> Result<(), Error> {
        if self.ncol != other.ncol {
            return Err(Error::shape_mismatch(self.ncol, other.ncol));
        }

        self.storage.data.extend_from_slice(other.storage.data);
        self.nrow += other.nrow;
        self.transposed = false;

        Ok(())
    }
}

impl<'a> Tensor<BF16, HostMemoryRef<'a>> {
    pub fn new(data: &'a [u8], shape: [usize; 2], is_row_major: bool) -> Result<Self, Error> {
        let volume = BF16::BYTES * shape[0] * shape[1];
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
        let volume = F32::BYTES * shape[0] * shape[1];
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

    #[allow(unused)]
    pub fn with_capacity(capacity: usize, shape: [usize; 2]) -> Result<Self, Error> {
        let data = Vec::with_capacity(capacity);
        let volume = F32::BYTES * shape[0] * shape[1];
        if volume != 0 {
            return Err(Error::shape_mismatch(volume, 0));
        }

        Ok(Self {
            storage: HostMemory { data },
            nrow: shape[0],
            ncol: shape[1],
            transposed: false,
            _phantom: PhantomData,
        })
    }
}

impl<'a> From<&Tensor<BF16, HostMemoryRef<'a>>> for Tensor<F32, HostMemory> {
    fn from(value: &Tensor<BF16, HostMemoryRef<'a>>) -> Self {
        let bytes = value.storage.data;
        let n = bytes.len() / BF16::BYTES;

        let mut data = vec![0; n * F32::BYTES];

        for i in 0..n {
            let hi = bytes[i * BF16::BYTES];
            let lo = bytes[i * BF16::BYTES + 1];
            data[i * F32::BYTES] = hi;
            data[i * F32::BYTES + 1] = lo;
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
