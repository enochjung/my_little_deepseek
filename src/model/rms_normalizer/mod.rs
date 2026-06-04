use super::{WeightInfo, build_tensor_f32};
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct RMSNormalizer<E: ElemType, L: Location>
where
    OwnConst: StorageType<'static, L>,
{
    norm_weight: Tensor<'static, OwnConst, E, L>,
}

impl RMSNormalizer<F32, Host> {
    pub(crate) fn new(weight_storage: &Mmap, norm_info: &WeightInfo) -> Result<Self, crate::Error> {
        let norm_weight = build_tensor_f32(weight_storage, norm_info)?;
        Ok(Self { norm_weight })
    }

    pub(crate) fn run_rms_norm(
        &self,
        x: &mut Tensor<'static, OwnMut, F32, Host>,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error> {
        x.rms_norm(&self.norm_weight, rms_norm_epsilon)
    }
}
