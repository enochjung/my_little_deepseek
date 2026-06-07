use crate::device::{DeviceOps, MutableDevice};
use crate::tensor::{ElemType, Tensor};

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

pub(crate) struct RoPE<E: ElemType, M: MutableDevice> {
    head_size: u32,
    cossin: Tensor<E, M>,
}

impl<E: ElemType, M: MutableDevice> RoPE<E, M>
where
    M::Base: DeviceOps<E>,
{
    pub(crate) fn new(
        tmp_1_x_d: Tensor<E, M>,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<Self, crate::Error> {
        let mut cossin = tmp_1_x_d;
        cossin.rope_vector(token_index, rope_theta, head_size)?;
        Ok(Self { head_size, cossin })
    }

    pub(crate) fn execute<M0: MutableDevice<Base = M::Base>, M1: MutableDevice<Base = M::Base>>(
        &self,
        x: &mut Tensor<E, M0>,
        tmp_1_x_d: &mut Tensor<E, M1>,
    ) -> Result<(), crate::Error> {
        let d = self.head_size;
        let half = d / 2;

        let cos = self.cossin.slice(0..1, 0..half);
        let sin = self.cossin.slice(0..1, half..d);

        tmp_1_x_d.copy(&x)?;
        let tmp0 = tmp_1_x_d.slice(0..1, 0..half);
        let tmp1 = tmp_1_x_d.slice(0..1, half..d);

        let (mut x0, mut x1) = x.slice_mut(0..1, 0..self.head_size).split_col(half)?;

        x0.mul_elementwise(&x1, &sin, -1.0)?;
        x1.mul_elementwise(&tmp1, &cos, 1.0)?;
        x0.mul_elementwise(&tmp0, &cos, 1.0)?;
        x1.mul_elementwise(&tmp0, &sin, 1.0)?;

        Ok(())
    }
}
