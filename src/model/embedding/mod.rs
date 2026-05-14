use super::WeightInfo;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Embedding<N: Numeric, S: Storage> {
    embed_tensor: TensorOwn<N, S>,
}

impl Embedding<F32, Host> {
    pub(crate) fn new(
        weight_storage: &Host,
        embed_tokens_weight: &WeightInfo,
    ) -> Result<Self, crate::Error> {
        let embed_tensor_bf16 =
            TensorRef::<BF16, Host>::try_from((weight_storage, embed_tokens_weight))?;
        let embed_tensor = TensorOwn::<F32, Host>::from(&embed_tensor_bf16);

        Ok(Self { embed_tensor })
    }

    pub(crate) fn word_embed(
        &self,
        token_id: u32,
    ) -> Result<TensorRef<'_, F32, Host>, crate::Error> {
        let ncol = self.embed_tensor.shape()[1];
        self.embed_tensor.as_ref(token_id..token_id + 1, 0..ncol)
    }
}
