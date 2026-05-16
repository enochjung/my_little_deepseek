use super::WeightInfo;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Embedding<N: Numeric, S: Storage> {
    embed_weight: TensorOwn<N, S>,
}

impl Embedding<F32, Host> {
    pub(crate) fn new(
        weight_storage: &Host,
        embed_info: &WeightInfo,
    ) -> Result<Self, crate::Error> {
        let embed_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            embed_info,
        ))?);

        Ok(Self { embed_weight })
    }

    pub(crate) fn word_embed(
        &self,
        token_id: u32,
    ) -> Result<TensorRef<'_, F32, Host>, crate::Error> {
        let ncol = self.embed_weight.shape()[1];
        self.embed_weight.as_ref(token_id..token_id + 1, 0..ncol)
    }
}
