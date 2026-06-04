use crate::storage::MmapMut;
use crate::tensor::*;

pub(crate) struct RoPE<E: ElemType, L: Location>
where
    Mut: StorageType<'static, L>,
    //Ref: StorageType<'static, L>,
{
    tensor_cs: Tensor<'static, Mut, E, L>,
    //tensor_cos: Tensor<'static, Ref, E, L>,
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

        let tensor_cos = tensor_cs.slice(0..1, 0..head_size / 2);

        Ok(Self {
            tensor_cs,
            //tensor_cos,
        })
    }

    // tmp: (1 * head_dim/2)
    pub(crate) fn run_rope(
        &self,
        x: &mut Tensor<'static, Mut, F32, Host>,
        tmp: &mut Tensor<'static, Mut, F32, Host>,
    ) -> Result<(), crate::Error> {
        // https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py#L116

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
