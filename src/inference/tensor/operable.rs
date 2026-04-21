use super::blas::{OpMM, OpVV};
use super::{DataType, Error, F32, HostMemory, Tensor};

#[allow(unused)]
const RMS_NORM_EPSILON_F32: f32 = 1e-6;

#[allow(unused)]
const ROPE_THETA: u32 = 10_000;

fn as_f32_slice(data: &[u8]) -> Result<&[f32], Error> {
    if data.len() % F32::BYTES != 0 {
        return Err(Error::shape_mismatch(0, data.len()));
    }

    let (prefix, values, suffix) = unsafe { data.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(Error::shape_mismatch(0, data.len()));
    }

    Ok(values)
}

fn as_f32_slice_mut(data: &mut [u8]) -> Result<&mut [f32], Error> {
    if data.len() % F32::BYTES != 0 {
        return Err(Error::shape_mismatch(0, data.len()));
    }

    let (prefix, values, suffix) = unsafe { data.align_to_mut::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(Error::shape_mismatch(
            0,
            prefix.len() + values.len() * 4 + suffix.len(),
        ));
    }

    Ok(values)
}

pub trait UnaryOperable {
    type Scalar;

    fn transpose(&mut self) -> ();
    fn silu(&mut self) -> ();
}

pub trait BinaryOperable<T> {
    type Scalar;

    fn muladd_weight_bias(&mut self, weight: &T, bias: &T) -> Result<(), Error>;
    fn rms_norm(&mut self, weight: &T) -> Result<(), Error>;
}

impl UnaryOperable for Tensor<F32, HostMemory> {
    type Scalar = f32;

    fn transpose(&mut self) -> () {
        self.is_trans = !self.is_trans;
    }

    fn silu(&mut self) -> () {
        let data = as_f32_slice_mut(&mut self.storage.data)
            .expect("Tensor<F32, HostMemory> must be aligned and sized for f32 operations");

        for v in data.iter_mut() {
            let x = *v;
            *v = x / (1.0 + (-x).exp());
        }
    }
}

impl BinaryOperable<Tensor<F32, HostMemory>> for Tensor<F32, HostMemory> {
    type Scalar = f32;

    fn muladd_weight_bias(
        &mut self,
        weight: &Tensor<F32, HostMemory>,
        bias: &Tensor<F32, HostMemory>,
    ) -> Result<(), Error> {
        if self.is_trans || weight.is_trans || bias.is_trans {
            return Err(Error::shape_mismatch(0, 1));
        }

        let n_tokens = self.nrow;
        let hidden_dim = self.ncol;
        if weight.nrow != hidden_dim {
            return Err(Error::shape_mismatch(hidden_dim, weight.nrow));
        }
        if weight.ncol != hidden_dim {
            return Err(Error::shape_mismatch(hidden_dim, weight.ncol));
        }
        if bias.nrow != 1 {
            return Err(Error::shape_mismatch(1, bias.nrow));
        }
        if bias.ncol != hidden_dim {
            return Err(Error::shape_mismatch(hidden_dim, bias.ncol));
        }

        let out = as_f32_slice_mut(&mut self.storage.data)?;
        let weight_slice = as_f32_slice(&weight.storage.data)?;
        let bias_slice = as_f32_slice(&bias.storage.data)?;

        OpMM::<F32, HostMemory>::muladd_mn_nn_1n(
            out,
            weight_slice,
            bias_slice,
            n_tokens,
            hidden_dim,
        );

        Ok(())
    }

    fn rms_norm(&mut self, weight: &Tensor<F32, HostMemory>) -> Result<(), Error> {
        let [nrow, ncol] = self.shape();
        let [wrow, wcol] = weight.shape();

        if wrow != 1 {
            return Err(Error::shape_mismatch(1, wrow));
        }
        if wcol != ncol {
            return Err(Error::shape_mismatch(ncol, wcol));
        }

        let data = as_f32_slice_mut(&mut self.storage.data)?;
        let wdata = as_f32_slice(&weight.storage.data)?;

        if data.len() != nrow * ncol {
            return Err(Error::shape_mismatch(nrow * ncol, data.len()));
        }
        if wdata.len() != ncol {
            return Err(Error::shape_mismatch(ncol, wdata.len()));
        }

        data.chunks_exact_mut(ncol).for_each(|row_data| {
            let rms = OpVV::<F32, HostMemory>::rms(row_data);
            let inv_rms = 1.0 / (rms + RMS_NORM_EPSILON_F32);
            OpVV::<F32, HostMemory>::mul(row_data, wdata, inv_rms);
        });

        Ok(())
    }
}
