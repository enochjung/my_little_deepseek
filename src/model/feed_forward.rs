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
        t: u32,
        target_t_x_h: &mut Tensor<E, M0>,
        tmp_3t_x_i: &mut Tensor<E, M1>,
    ) -> Result<(), crate::Error> {
        let i = self.intermediate_size;

        let tmp_3t_x_i = tmp_3t_x_i.slice_mut(0..t * 3, 0..i);
        let (mut gate_t_x_i, tmp_2t_x_i) = tmp_3t_x_i.split_row(t)?;
        let (mut up_t_x_i, mut activated_t_x_i) = tmp_2t_x_i.split_row(t)?;

        self.rms_norm.execute(target_t_x_h)?;

        gate_t_x_i.mul_bt(&target_t_x_h, &self.gate.transpose())?;
        up_t_x_i.mul_bt(&target_t_x_h, &self.up.transpose())?;

        gate_t_x_i.silu();

        activated_t_x_i.mul_elementwise(&gate_t_x_i, &up_t_x_i, 1.0)?;

        target_t_x_h.mul_bt(&activated_t_x_i, &self.down.transpose())?;

        Ok(())
    }
}
