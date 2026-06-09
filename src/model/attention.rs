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
        t: u32,
        target_t_x_h: &mut Tensor<E, M0>,
        tmp_2_x_d: &mut Tensor<E, M1>,
        tmp_t_x_h: &mut Tensor<E, M2>,
        tmp_t_x_nt: &mut Tensor<E, M3>,
    ) -> Result<(), crate::Error>
    where
        D: Device<Base = OD>,
    {
        let d = self.head_size;
        let h = d * self.num_attention_heads as u32;
        let n = kv_cache.n() as u32;
        let nt = n + t;
        let kvd = d * self.num_key_value_heads as u32;

        let qtmp_t_x_h = tmp_t_x_h;
        let (mut ktmp_t_x_kvd, mut vtmp_t_x_kvd) = kv_cache.allocate(t)?;
        let tmp_2_x_d = tmp_2_x_d.slice_mut(0..2, 0..d);
        let (mut ropetmp0_1_x_d, mut ropetmp1_1_x_d) = tmp_2_x_d.split_row(1)?;

        self.rms_norm.execute(target_t_x_h)?;

        qtmp_t_x_h.muladd_bt_broadcast(&target_t_x_h, &self.q_weight.transpose(), &self.q_bias)?;
        ktmp_t_x_kvd.muladd_bt_broadcast(
            &target_t_x_h,
            &self.k_weight.transpose(),
            &self.k_bias,
        )?;
        vtmp_t_x_kvd.muladd_bt_broadcast(
            &target_t_x_h,
            &self.v_weight.transpose(),
            &self.v_bias,
        )?;

        for i in 0..t {
            let rope = RoPE::new(&mut ropetmp0_1_x_d, n + i, self.rope_theta, d)?;

            let mut qtarget_1_x_h = qtmp_t_x_h.slice_mut(i..i + 1, 0..h);
            let mut ktarget_1_x_h = ktmp_t_x_kvd.slice_mut(i..i + 1, 0..kvd);

            rope.execute(
                &mut qtarget_1_x_h,
                &mut ropetmp1_1_x_d,
                self.num_attention_heads,
            )?;
            rope.execute(
                &mut ktarget_1_x_h,
                &mut ropetmp1_1_x_d,
                self.num_key_value_heads,
            )?;
        }

        let score_t_x_nt = tmp_t_x_nt;

        let (kref_nt_x_kvd, vref_nt_x_kvd) = kv_cache.get_kv();
        for i in 0..self.num_attention_heads as u32 {
            let kvi = i / (self.num_attention_heads / self.num_key_value_heads) as u32;

            let mut qi_t_x_d = qtmp_t_x_h.slice_mut(0..t, i * d..(i + 1) * d);
            let ki_nt_x_d = kref_nt_x_kvd.slice(0..nt, kvi * d..(kvi + 1) * d);
            let vi_nt_x_d = vref_nt_x_kvd.slice(0..nt, kvi * d..(kvi + 1) * d);

            score_t_x_nt.mul_bt(&qi_t_x_d, &ki_nt_x_d.transpose())?;
            score_t_x_nt.safe_softmax_with_masking(1.0 / ((d as f32).sqrt()));
            qi_t_x_d.mul(&score_t_x_nt, &vi_nt_x_d)?;
        }

        target_t_x_h.mul_bt(&qtmp_t_x_h, &self.o_weight.transpose())?;

        Ok(())
    }
}
