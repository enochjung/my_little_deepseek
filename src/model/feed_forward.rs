use super::FeedForwardLayerWeightInfo;
use super::{RMSNorm, build_casted_tensor};
use crate::device::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::tensor::{ElemType, F32, Tensor};

pub(crate) struct FeedForward<E: ElemType, D: Device> {
    rms_norm: RMSNorm<E, D>,
    gate: Tensor<E, D>,
    up: Tensor<E, D>,
    down: Tensor<E, D>,

    intermediate_size: u32,
}

impl<OD: OwnedDevice> FeedForward<F32, OD> {
    pub(crate) fn new(
        weight_storage: &OD,
        feed_forward_weight_info: &FeedForwardLayerWeightInfo,
        intermediate_size: u32,
        rms_norm_epsilon: f32,
    ) -> Result<Self, crate::Error>
    where
        OD: DeviceOps<F32>,
    {
        let gate = build_casted_tensor(weight_storage, &feed_forward_weight_info.gate_proj_weight)?;
        let up = build_casted_tensor(weight_storage, &feed_forward_weight_info.up_proj_weight)?;
        let down = build_casted_tensor(weight_storage, &feed_forward_weight_info.down_proj_weight)?;

        let rms_norm = RMSNorm::new(
            weight_storage,
            &feed_forward_weight_info.post_attention_layernorm_weight,
            rms_norm_epsilon,
        )?;

        Ok(Self {
            rms_norm,
            gate,
            up,
            down,

            intermediate_size,
        })
    }
}

impl<E: ElemType, D: Device> FeedForward<E, D>
where
    D::Base: DeviceOps<E>,
{
    pub(crate) fn execute<M0: MutableDevice<Base = D::Base>, M1: MutableDevice<Base = D::Base>>(
        &self,
        x: &mut Tensor<E, M0>,
        tmp_3_x_i: &mut Tensor<E, M1>,
    ) -> Result<(), crate::Error> {
        let i = self.intermediate_size;

        let tmp_3_x_i = tmp_3_x_i.slice_mut(0..2, 0..i);
        let (mut gate, tmp_2_x_i) = tmp_3_x_i.split_row(1)?;
        let (mut up, mut activated) = tmp_2_x_i.split_row(1)?;

        self.rms_norm.execute(x)?;

        gate.mul_bt(&x, &self.gate.transpose())?;
        up.mul_bt(&x, &self.up.transpose())?;

        gate.silu();

        activated.mul_elementwise(&gate, &up, 1.0)?;

        x.mul_bt(&activated, &self.down.transpose())?;

        Ok(())
    }
}
