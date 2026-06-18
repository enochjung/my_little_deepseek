use core::{ElemType, MLTError, MemoryMut};

use crate::tensor::Tensor;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

pub struct RoPE<'a, T: ElemType, MM: MemoryMut<T>> {
    cossin: &'a Tensor<T, MM>,
    head_size: u32,
}

impl<'a, T: ElemType, MM: MemoryMut<T>> RoPE<'a, T, MM> {
    pub fn new(
        tmp_1_x_d: &'a mut Tensor<T, MM>,
        token_index: u32,
        rope_theta: T,
        head_size: u32,
    ) -> Result<Self, MLTError> {
        let cossin = tmp_1_x_d;
        cossin.rope_vector(token_index, rope_theta, head_size)?;
        Ok(Self { head_size, cossin })
    }

    pub fn execute<D: MemoryMut<T, Base = MM::Base>, T0: MemoryMut<T, Base = MM::Base>>(
        &self,
        target_1_x_ad: &mut Tensor<T, D>,
        tmp_1_x_d: &mut Tensor<T, T0>,
        iter: usize,
    ) -> Result<(), MLTError> {
        let d = self.head_size;
        let half = d / 2;

        let cos = self.cossin.slice(0..1, 0..half);
        let sin = self.cossin.slice(0..1, half..d);

        for i in 0..iter as u32 {
            let mut dst = target_1_x_ad.slice_mut(0..1, i * d..(i + 1) * d);
            tmp_1_x_d.copy(&dst)?;

            let (mut x0, mut x1) = dst.split_col(half)?;
            let (tmp0, tmp1) = tmp_1_x_d.split_col(half)?;

            x0.elem_mul_assign(&cos)?;
            x1.elem_mul_assign(&cos)?;
            x0.elem_mulsub_assign(&tmp1, &sin)?;
            x1.elem_muladd_assign(&tmp0, &sin)?;
        }

        Ok(())
    }
}
