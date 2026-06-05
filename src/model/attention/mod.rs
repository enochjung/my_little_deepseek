use super::{AttentionLayerWeightInfo, RoPE};
use super::{RMSNormalizer, build_tensor_f32};
use crate::session::KVCache;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Attention<E: ElemType, O: Owned> {
    q_bias: Tensor<E, O>,
    q_weight: Tensor<E, O>,
    k_bias: Tensor<E, O>,
    k_weight: Tensor<E, O>,
    v_bias: Tensor<E, O>,
    v_weight: Tensor<E, O>,
    o_weight: Tensor<E, O>,
    rms_normalizer: RMSNormalizer<E, O>,
}

impl Attention<F32, Mmap> {
    pub(crate) fn new(
        weight_storage: &Mmap,
        attention_weight_info: &AttentionLayerWeightInfo,
    ) -> Result<Self, crate::Error> {
        let q_bias = build_tensor_f32(weight_storage, &attention_weight_info.q_proj_bias)?;
        let q_weight = build_tensor_f32(weight_storage, &attention_weight_info.q_proj_weight)?;
        let k_bias = build_tensor_f32(weight_storage, &attention_weight_info.k_proj_bias)?;
        let k_weight = build_tensor_f32(weight_storage, &attention_weight_info.k_proj_weight)?;
        let v_bias = build_tensor_f32(weight_storage, &attention_weight_info.v_proj_bias)?;
        let v_weight = build_tensor_f32(weight_storage, &attention_weight_info.v_proj_weight)?;
        let o_weight = build_tensor_f32(weight_storage, &attention_weight_info.o_proj_weight)?;

        let rms_normalizer = RMSNormalizer::new(
            weight_storage,
            &attention_weight_info.input_layernorm_weight,
        )?;

        Ok(Self {
            q_bias,
            q_weight,
            k_bias,
            k_weight,
            v_bias,
            v_weight,
            o_weight,
            rms_normalizer,
        })
    }

    pub(crate) fn run_attention(
        &self,
        x: &mut Tensor<F32, MmapMut>,
        kv_cache: &mut KVCache<F32, MmapMut>,

        num_attention_heads: usize,
        num_key_value_heads: usize,
        rms_norm_epsilon: f32,
        head_size: u32,
        rope_theta: f32,
    ) -> Result<(), crate::Error> {
        let cs_storage = MmapMut::new(1 * head_size as usize * F32::BYTES)?;
        let n = kv_cache.len() as u32;
        let rope_engine = RoPE::new(cs_storage, n, rope_theta, head_size)?;
        let mut rope_tmp = Tensor::<F32, MmapMut>::new(
            MmapMut::new(1 * head_size as usize * F32::BYTES)?,
            0,
            1,
            head_size,
            head_size,
            true,
        )?;

        let mut q = Tensor::<F32, MmapMut>::new(
            MmapMut::new(1 * head_size as usize * F32::BYTES)?,
            0,
            1,
            head_size,
            head_size,
            true,
        )?;

        let mut score = Tensor::<F32, MmapMut>::new(
            MmapMut::new(1 * (n + 1) as usize * F32::BYTES)?,
            0,
            1,
            n + 1,
            n + 1,
            true,
        )?;

        self.rms_normalizer.run_rms_norm(x, rms_norm_epsilon)?;

        let (mut k, mut v) = kv_cache.allocate()?;
        q.muladd_broadcast(&x, &self.q_weight, &self.q_bias)?;
        k.muladd_broadcast(&x, &self.k_weight, &self.k_bias)?;
        v.muladd_broadcast(&x, &self.v_weight, &self.v_bias)?;
        rope_engine.run_rope(&mut q, &mut rope_tmp)?;
        rope_engine.run_rope(&mut k, &mut rope_tmp)?;

        let (k, v) = kv_cache.get_kv();
        let k = k.transpose();
        let v = v.transpose();
        for i in 0..num_attention_heads as u32 {
            let kvi = i / (num_attention_heads / num_key_value_heads) as u32;

            let mut qi = q.slice_mut(0..1, i * head_size..(i + 1) * head_size);
            let ki = k.slice(kvi * head_size..(kvi + 1) * head_size, 0..n + 1);
            let vi = v.slice(kvi * head_size..(kvi + 1) * head_size, 0..n + 1);

            score.mul(&qi, &ki)?;
            score.softmax(1.0 / (head_size as f32).sqrt())?;
            qi.mul(&score, &vi)?;
        }

        x.mul(&q, &self.o_weight)?;

        Ok(())
    }
}
