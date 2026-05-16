use super::WeightInfo;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct RMSNormalizer<N: Numeric, S: Storage> {
    norm_weight: TensorOwn<N, S>,
}

impl RMSNormalizer<F32, Host> {
    pub(crate) fn new(
        weight_storage: &Host,
        norm_weight_info: &WeightInfo,
    ) -> Result<Self, crate::Error> {
        let norm_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            norm_weight_info,
        ))?);

        Ok(Self { norm_weight })
    }

    pub(crate) fn apply_rms_norm(
        &self,
        target: &mut TensorOwn<F32, Host>,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error> {
        target.rms_norm(&self.norm_weight, rms_norm_epsilon)
    }
}
