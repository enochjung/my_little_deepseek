use crate::device::{DeviceOps, MutableDevice};
use crate::tensor::{ElemType, Tensor};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

pub(crate) struct RoPE<'a, E: ElemType, M: MutableDevice> {
    head_size: u32,
    cossin: &'a mut Tensor<E, M>,
}

impl<'a, E: ElemType, M: MutableDevice> RoPE<'a, E, M>
where
    M::Base: DeviceOps<E>,
{
    pub(crate) fn new(
        tmp_1_x_d: &'a mut Tensor<E, M>,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<Self, crate::Error> {
        let cossin = tmp_1_x_d;
        cossin.rope_vector(token_index, rope_theta, head_size)?;
        Ok(Self { head_size, cossin })
    }

    pub(crate) fn execute<M0: MutableDevice<Base = M::Base>, M1: MutableDevice<Base = M::Base>>(
        &self,
        target_1_x_ad: &mut Tensor<E, M0>,
        tmp_1_x_d: &mut Tensor<E, M1>,
        iter: usize,
    ) -> Result<(), crate::Error> {
        let d = self.head_size;
        let half = d / 2;

        let cos = self.cossin.slice(0..1, 0..half);
        let sin = self.cossin.slice(0..1, half..d);

        for i in 0..iter as u32 {
            let dst = target_1_x_ad.slice_mut(0..1, i * d..(i + 1) * d);

            tmp_1_x_d.copy(&dst)?;
            let tmp_1_x_d = tmp_1_x_d.slice_mut(0..1, 0..d);
            let (mut x0, mut x1) = dst.split_col(half)?;
            let (mut tmp0, tmp1) = tmp_1_x_d.split_col(half)?;

            x0.mul_elementwise(&tmp0, &cos, 1.0)?;
            x1.mul_elementwise(&tmp0, &sin, 1.0)?;
            tmp0.mul_elementwise(&tmp1, &sin, -1.0)?;
            x0.add(&tmp0)?;
            tmp0.mul_elementwise(&tmp1, &cos, 1.0)?;
            x1.add(&tmp0)?;
        }

        Ok(())
    }
}
