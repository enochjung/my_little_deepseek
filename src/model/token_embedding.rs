use super::{WeightInfo, build_casted_tensor};
use crate::device::{Device, DeviceOps, OwnedDevice};
use crate::tensor::{ElemType, F32, Tensor};

pub(crate) struct TokenEmbedding<E: ElemType, D: Device> {
    embed_weight: Tensor<E, D>,
    vocab_size: u32,
    hidden_size: u32,
}

impl<OD: OwnedDevice> TokenEmbedding<F32, OD> {
    pub(crate) fn new(weight_storage: &OD, embed_info: &WeightInfo) -> Result<Self, crate::Error>
    where
        OD: DeviceOps<F32>,
    {
        let vocab_size = embed_info.shape[0];
        let hidden_size = embed_info.shape[1];
        let embed_weight = build_casted_tensor(weight_storage, embed_info)?;

        Ok(Self {
            embed_weight,
            vocab_size,
            hidden_size,
        })
    }
}

impl<E: ElemType, D: Device> TokenEmbedding<E, D> {
    pub(crate) fn execute(&self, token_id: u32) -> Result<Tensor<E, &D::Base>, crate::Error> {
        if token_id >= self.vocab_size {
            return Err(crate::Error::out_of_bound(
                token_id as usize,
                self.vocab_size as usize,
            ));
        }
        Ok(self
            .embed_weight
            .slice(token_id..token_id + 1, 0..self.hidden_size))
    }
}
