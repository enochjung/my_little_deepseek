use super::{WeightInfo, build_tensor_f32};
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct RMSNormalizer<'a, E: ElemType, L: Location>
where
    Own: StorageType<'a, L>,
{
    norm_weight: Tensor<'a, Own, E, L>,
}

impl RMSNormalizer<'_, F32, Host> {
    pub(crate) fn new(weight_storage: &Mmap, norm_info: &WeightInfo) -> Result<Self, crate::Error> {
        let norm_weight = build_tensor_f32(weight_storage, norm_info)?;
        Ok(Self { norm_weight })
    }

    pub(crate) fn apply_rms_norm<'a>(
        &self,
        target: &mut Tensor<'a, Mut, F32, Host>,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error>
    where
        Mut: StorageType<'a, Host>,
    {
        target.rms_norm(&self.norm_weight, rms_norm_epsilon)
    }
}
