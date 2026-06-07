use super::{WeightInfo, build_casted_tensor};
use crate::device::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::tensor::{ElemType, F32, Tensor};

pub(crate) struct RMSNorm<E: ElemType, D: Device> {
    norm_weight: Tensor<E, D>,
    epsilon: f32,
}

impl<OD: OwnedDevice> RMSNorm<F32, OD> {
    pub(crate) fn new(
        weight_storage: &OD,
        norm_info: &WeightInfo,
        epsilon: f32,
    ) -> Result<Self, crate::Error>
    where
        OD: DeviceOps<F32>,
    {
        let norm_weight = build_casted_tensor(weight_storage, norm_info)?;
        Ok(Self {
            norm_weight,
            epsilon,
        })
    }
}

impl<E: ElemType, D: Device> RMSNorm<E, D>
where
    D::Base: DeviceOps<E>,
{
    pub(crate) fn execute<M0: MutableDevice<Base = D::Base>>(
        &self,
        x: &mut Tensor<E, M0>,
    ) -> Result<(), crate::Error> {
        x.rms_norm(&self.norm_weight, self.epsilon)
    }
}
