use super::{WeightInfo, build_tensor_f32};
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Embedding<E: ElemType, O: Owned> {
    embed_weight: Tensor<E, O>,
    nrow: u32,
    ncol: u32,
}

impl Embedding<F32, Mmap> {
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

    pub(crate) fn word_embed(&self, token_id: u32) -> Result<Tensor<F32, &Mmap>, crate::Error> {
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
