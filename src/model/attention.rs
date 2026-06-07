use super::{AttentionLayerWeightInfo, RoPE};
use super::{RMSNorm, build_casted_tensor};
use crate::device::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::session::KVCache;
use crate::tensor::{ElemType, F32, Tensor};

pub(crate) struct Attention<E: ElemType, D: Device> {
    rms_norm: RMSNorm<E, D>,
    q_bias: Tensor<E, D>,
    q_weight: Tensor<E, D>,
    k_bias: Tensor<E, D>,
    k_weight: Tensor<E, D>,
    v_bias: Tensor<E, D>,
    v_weight: Tensor<E, D>,
    o_weight: Tensor<E, D>,

    head_size: u32,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: f32,
}

impl<OD: OwnedDevice> Attention<F32, OD> {
    pub(crate) fn new(
        weight_storage: &OD,
        attention_weight_info: &AttentionLayerWeightInfo,
        head_size: u32,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rms_norm_epsilon: f32,
        rope_theta: f32,
    ) -> Result<Self, crate::Error>
    where
        OD: DeviceOps<F32>,
    {
        let q_bias = build_casted_tensor(weight_storage, &attention_weight_info.q_proj_bias)?;
        let q_weight = build_casted_tensor(weight_storage, &attention_weight_info.q_proj_weight)?;
        let k_bias = build_casted_tensor(weight_storage, &attention_weight_info.k_proj_bias)?;
        let k_weight = build_casted_tensor(weight_storage, &attention_weight_info.k_proj_weight)?;
        let v_bias = build_casted_tensor(weight_storage, &attention_weight_info.v_proj_bias)?;
        let v_weight = build_casted_tensor(weight_storage, &attention_weight_info.v_proj_weight)?;
        let o_weight = build_casted_tensor(weight_storage, &attention_weight_info.o_proj_weight)?;

        let rms_norm = RMSNorm::new(
            weight_storage,
            &attention_weight_info.input_layernorm_weight,
            rms_norm_epsilon,
        )?;

        Ok(Self {
            rms_norm,
            q_bias,
            q_weight,
            k_bias,
            k_weight,
            v_bias,
            v_weight,
            o_weight,

            head_size,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
        })
    }
}

impl<E: ElemType, D: Device> Attention<E, D>
where
    D::Base: DeviceOps<E>,
{
    pub(crate) fn execute<
        OD: OwnedDevice,
        M0: MutableDevice<Base = D::Base>,
        M1: MutableDevice<Base = D::Base>,
        M2: MutableDevice<Base = D::Base>,
        M3: MutableDevice<Base = D::Base>,
    >(
        &self,
        kv_cache: &mut KVCache<E, OD>,
        x: &mut Tensor<E, M0>,
        tmp_2_x_d: &mut Tensor<E, M1>,
        tmp_1_x_h: &mut Tensor<E, M2>,
        tmp_1_x_n1: &mut Tensor<E, M3>,
    ) -> Result<(), crate::Error>
    where
        D: Device<Base = OD>,
    {
        let d = self.head_size;
        let n = kv_cache.n() as u32;

        let tmp_2_x_d = tmp_2_x_d.slice_mut(0..2, 0..d);
        let (tmp0_1_x_d, mut tmp1_1_x_d) = tmp_2_x_d.split_row(1)?;
        let rope = RoPE::new(tmp0_1_x_d, n, self.rope_theta, d)?;
        let q = tmp_1_x_h;
        let score = tmp_1_x_n1;

        self.rms_norm.execute(x)?;

        let (mut k, mut v) = kv_cache.allocate()?;
        q.muladd_bt_broadcast(&x, &self.q_weight.transpose(), &self.q_bias)?;
        k.muladd_bt_broadcast(&x, &self.k_weight.transpose(), &self.k_bias)?;
        v.muladd_bt_broadcast(&x, &self.v_weight.transpose(), &self.v_bias)?;
        rope.execute(q, &mut tmp1_1_x_d)?;
        rope.execute(&mut k, &mut tmp1_1_x_d)?;

        let (k, v) = kv_cache.get_kv();
        let k = k;
        let v = v;
        for i in 0..self.num_attention_heads as u32 {
            let kvi = i / (self.num_attention_heads / self.num_key_value_heads) as u32;

            let mut qi = q.slice_mut(0..1, i * d..(i + 1) * d);
            let ki = k.slice(0..n + 1, kvi * d..(kvi + 1) * d);
            let vi = v.slice(0..n + 1, kvi * d..(kvi + 1) * d);

            score.mul_bt(&qi, &ki.transpose())?;
            score.softmax(1.0 / (d as f32).sqrt());
            qi.mul_bt(&score, &vi.transpose())?;
        }

        x.mul(&q, &self.o_weight)?;

        Ok(())
    }
}
