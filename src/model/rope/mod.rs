use crate::storage::MmapMut;
use crate::tensor::*;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

pub(crate) struct RoPE<E: ElemType, L: Location>
where
    Mut: StorageType<'static, L>,
{
    head_size: u32,
    tensor_cs: Tensor<'static, Mut, E, L>,
}

impl RoPE<F32, Host> {
    pub(crate) fn new(
        data: MmapMut,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<Self, crate::Error> {
        // build cos & sin vec.

        // build tensor_cs with data.
        // tensor_cs size : (1 * head_size)

        // cs_tensor[i] = {
        //     cos(token_index / (rope_theta ^ (2 * i / head_size)))  if i < head_size/2;
        //     sin(token_index / (rope_theta ^ (2 * (i-head_size/2) / head_size)))  if head_size/2 <= i;
        // }
        // tensor::rope_cs makes like that.

        // store tensor_cos and tensor_sin with tensor::slice

        // Create a mutable tensor backed by the provided MmapMut, fill it with
        // RoPE cos/sin values using the tensor helper, then convert it to an
        // owned (readonly) tensor for safe storage.
        let mut tensor_cs = Tensor::<Mut, F32, Host>::new(data, 1, head_size, true)?;
        tensor_cs.rope_cs(token_index, rope_theta, head_size)?;

        Ok(Self {
            head_size,
            tensor_cs,
        })
    }

    // tmp: (1 * head_dim/2)
    pub(crate) fn run_rope(
        &self,
        x: &mut Tensor<'static, Mut, F32, Host>,
        tmp: &mut Tensor<'static, Mut, F32, Host>,
    ) -> Result<(), crate::Error> {
        let n = self.head_size;
        let half = n / 2;

        let tensor_cos = self.tensor_cs.slice(0..1, 0..half);
        let tensor_sin = self.tensor_cs.slice(0..1, half..n);

        // x_0 = x[0..half]
        // x_1 = x[half..head_size]

        // clone x_0 to tmp.
        // x_0 = -1 * x_1 ** sin.
        // x_1 = x_1 ** cos.
        // x_0 += tmp ** cos.
        // x_1 += tmp ** sin.
        todo!()
    }

    pub(crate) fn take_storage(self) -> () {
        // return cos & sin data
        todo!()
    }
}
