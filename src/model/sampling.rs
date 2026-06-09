use super::WeightInfo;
use super::{RMSNorm, build_casted_tensor};
use crate::device::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::tensor::{ElemType, F32, Tensor};

pub(crate) struct Sampling<E: ElemType, D: Device> {
    rms_norm: RMSNorm<E, D>,
    lm_head: Tensor<E, D>,
}

impl<OD: OwnedDevice> Sampling<F32, OD> {
    pub(crate) fn new(
        weight_storage: &OD,
        norm_weight: &WeightInfo,
        lm_head_weight: &WeightInfo,
        rms_norm_epsilon: f32,
    ) -> Result<Self, crate::Error>
    where
        OD: DeviceOps<F32>,
    {
        let lm_head = build_casted_tensor(weight_storage, lm_head_weight)?;
        let rms_norm = RMSNorm::new(weight_storage, &norm_weight, rms_norm_epsilon)?;

        Ok(Self { rms_norm, lm_head })
    }
}

impl<E: ElemType, D: Device> Sampling<E, D>
where
    D::Base: DeviceOps<E>,
{
    pub(crate) fn execute<M0: MutableDevice<Base = D::Base>, M1: MutableDevice<Base = D::Base>>(
        &self,
        target_1_x_h: Tensor<E, M0>,
        tmp_1_x_v: &mut Tensor<E, M1>,
    ) -> Result<u32, crate::Error> {
        let mut target_1_x_h = target_1_x_h;

        self.rms_norm.execute(&mut target_1_x_h)?;
        tmp_1_x_v.mul_bt(&target_1_x_h, &self.lm_head.transpose())?;

        let next_token = tmp_1_x_v.argmax()?;
        Ok(next_token)
    }
}
