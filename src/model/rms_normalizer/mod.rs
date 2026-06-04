use super::{WeightInfo, build_tensor_f32};
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct RMSNormalizer<E: ElemType, O: Owned> {
    norm_weight: Tensor<E, O>,
}

impl RMSNormalizer<F32, Mmap> {
    pub(crate) fn new(weight_storage: &Mmap, norm_info: &WeightInfo) -> Result<Self, crate::Error> {
        let norm_weight = build_tensor_f32(weight_storage, norm_info)?;
        Ok(Self { norm_weight })
    }

    pub(crate) fn run_rms_norm(
        &self,
        x: &mut Tensor<F32, MmapMut>,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error> {
        x.rms_norm(&self.norm_weight, rms_norm_epsilon)
    }
}
