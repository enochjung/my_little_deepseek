use crate::storage::*;
use crate::tensor::*;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

pub(crate) struct RoPE<E: ElemType, MO: Mutable + Owned> {
    head_size: u32,
    tensor_cs: Tensor<E, MO>,
}

impl RoPE<F32, MmapMut> {
    pub(crate) fn new(
        data: MmapMut,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<Self, crate::Error> {
        let mut tensor_cs = Tensor::<F32, _>::new(data, 0, 1, head_size, head_size, true)?;
        tensor_cs.rope_cs(token_index, rope_theta, head_size)?;

        Ok(Self {
            head_size,
            tensor_cs,
        })
    }

    // tmp: (1 * head_dim)
    pub(crate) fn run_rope(
        &self,
        x: &mut Tensor<F32, MmapMut>,
        tmp: &mut Tensor<F32, MmapMut>,
    ) -> Result<(), crate::Error> {
        let n = self.head_size;
        let half = n / 2;

        let tensor_cos = self.tensor_cs.slice(0..1, 0..half);
        let tensor_sin = self.tensor_cs.slice(0..1, half..n);

        tmp.copy(&x)?;
        let tmp0 = tmp.slice(0..1, 0..half);
        let tmp1 = tmp.slice(0..1, half..n);

        let (mut x0, mut x1) = x.slice_mut(0..1, 0..self.head_size).split_col(half)?;

        x0.mul_elementwise(&x1, &tensor_sin, -1.0)?;
        x1.mul_elementwise(&tmp1, &tensor_cos, 1.0)?;
        x0.mul_elementwise(&tmp0, &tensor_cos, 1.0)?;
        x1.mul_elementwise(&tmp0, &tensor_sin, 1.0)?;

        Ok(())
    }

    pub(crate) fn take_storage(self) -> MmapMut {
        self.tensor_cs.take_storage()
    }
}
