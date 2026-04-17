use super::{DataType, Error, F32, HostMemory, Tensor};

#[allow(unused)]
const RMS_NORM_EPSILON_F32: f32 = 1e-6;

#[allow(unused)]
const ROPE_THETA: u32 = 10_000;

#[allow(unused)]
pub trait UnaryOperable {
    type Scalar;

    fn transpose(&mut self) -> ();
    fn mul(&mut self, value: Self::Scalar) -> ();
    fn silu(&mut self) -> ();
    fn rms(&self) -> Self::Scalar;
}

#[allow(unused)]
pub trait BinaryOperable<T> {
    type Scalar;

    fn matmul(&mut self, rhs: &T) -> Result<(), Error>;
    fn mul(&mut self, rhs: &T) -> Result<(), Error>;
    fn add(&mut self, rhs: &T) -> Result<(), Error>;
    fn rms_norm(&mut self, weight: &T) -> Result<(), Error>;
}

impl UnaryOperable for Tensor<F32, HostMemory> {
    type Scalar = f32;

    fn transpose(&mut self) -> () {
        let new_nrow = self.ncol;
        let new_ncol = self.nrow;

        self.nrow = new_nrow;
        self.ncol = new_ncol;
        self.transposed = !self.transposed;
    }

    fn mul(&mut self, _value: Self::Scalar) -> () {
        todo!()
    }

    fn silu(&mut self) -> () {
        let n = self.storage.data.len() / F32::BYTES;
        let ptr = self.storage.data.as_mut_ptr().cast::<f32>();
        let data = unsafe { core::slice::from_raw_parts_mut(ptr, n) };

        for v in data.iter_mut() {
            let x = *v;
            *v = x / (1.0 + (-x).exp());
        }
    }

    fn rms(&self) -> Self::Scalar {
        todo!()
    }
}

impl BinaryOperable<Tensor<F32, HostMemory>> for Tensor<F32, HostMemory> {
    type Scalar = f32;

    fn matmul(&mut self, _rhs: &Tensor<F32, HostMemory>) -> Result<(), Error> {
        todo!()
    }

    fn mul(&mut self, _rhs: &Tensor<F32, HostMemory>) -> Result<(), Error> {
        todo!()
    }

    fn add(&mut self, _rhs: &Tensor<F32, HostMemory>) -> Result<(), Error> {
        todo!()
    }

    fn rms_norm(&mut self, _weight: &Tensor<F32, HostMemory>) -> Result<(), Error> {
        todo!()
    }
}
