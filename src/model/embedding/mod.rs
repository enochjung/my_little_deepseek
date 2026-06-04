use super::{WeightInfo, build_tensor_f32};
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Embedding<E: ElemType, L: Location>
where
    OwnConst: StorageType<'static, L>,
{
    embed_weight: Tensor<'static, OwnConst, E, L>,
    nrow: u32,
    ncol: u32,
}

impl Embedding<F32, Host> {
    pub(crate) fn new(
        weight_storage: &Mmap,
        embed_info: &WeightInfo,
    ) -> Result<Self, crate::Error> {
        let nrow = embed_info.shape[0];
        let ncol = embed_info.shape[1];
        let embed_weight = build_tensor_f32(weight_storage, embed_info)?;
        Ok(Self {
            embed_weight,
            nrow,
            ncol,
        })
    }

    pub(crate) fn word_embed<'a>(
        &'a self,
        token_id: u32,
    ) -> Result<Tensor<'a, RefConst, F32, Host>, crate::Error> {
        if token_id >= self.nrow {
            return Err(crate::Error::out_of_bound(
                token_id as usize,
                self.nrow as usize,
            ));
        }
        Ok(self
            .embed_weight
            .slice(token_id..token_id + 1, 0..self.ncol))
    }
}
