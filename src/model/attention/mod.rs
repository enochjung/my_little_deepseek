use super::AttentionLayerWeightInfo;
use super::{RMSNormalizer, build_tensor_f32};
use crate::session::KVCache;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Attention<E: ElemType, L: Location>
where
    OwnConst: StorageType<'static, L>,
{
    q_bias: Tensor<'static, OwnConst, E, L>,
    q_weight: Tensor<'static, OwnConst, E, L>,
    k_bias: Tensor<'static, OwnConst, E, L>,
    k_weight: Tensor<'static, OwnConst, E, L>,
    v_bias: Tensor<'static, OwnConst, E, L>,
    v_weight: Tensor<'static, OwnConst, E, L>,
    o_weight: Tensor<'static, OwnConst, E, L>,
    rms_normalizer: RMSNormalizer<E, L>,
}

impl Attention<F32, Host> {
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
        x: &mut Tensor<'static, OwnMut, F32, Host>,
        kv_cache: &mut KVCache,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error> {
        self.rms_normalizer.run_rms_norm(x, rms_norm_epsilon)?;

        todo!()
        /*
              // ① Pre-RMS Norm

              // ② Q, K, V Projection
              // [Shape] q_proj: [1, 1536] (12 heads * 128)
              // [Shape] k_proj: [1, 256]  (2 heads * 128)
              // [Shape] v_proj: [1, 256]  (2 heads * 128)
              let q_proj = matmul(&norm_x, &weights.layers[layer].q_proj);
              let k_proj = matmul(&norm_x, &weights.layers[layer].k_proj);
              let v_proj = matmul(&norm_x, &weights.layers[layer].v_proj);

              // ③ 헤드별로 분할 (Reshape)
              // [Shape] q: [12, 1, 128]
              // [Shape] k: [2, 1, 128]
              // [Shape] v: [2, 1, 128]
              let mut q = reshape_to_heads::<NUM_Q_HEADS>(&q_proj);
              let mut k = reshape_to_heads::<NUM_KV_HEADS>(&k_proj);
              let v = reshape_to_heads::<NUM_KV_HEADS>(&v_proj);

              // ④ RoPE (Rotary Position Embedding) 적용
              // 주의: RoPE는 Q와 K에만 적용되며, V에는 적용되지 않습니다!
              // 현재 위치 `current_pos`(N)에 해당하는 회전 변환만 수행합니다.
              // [Shape] q, k 크기 변동 없음
              apply_rope_in_place(&mut q, current_pos);
              apply_rope_in_place(&mut k, current_pos);

              // ⑤ KV Cache 업데이트 및 가져오기 (가장 헷갈려하시는 부분)
              // 현재 스텝의 K, V를 캐시에 '추가(Append)' 합니다.
              kv_cache[layer].k_cache.append(k); // 이전 크기 [2, N, 128] -> [2, N+1, 128]
              kv_cache[layer].v_cache.append(v); // 이전 크기 [2, N, 128] -> [2, N+1, 128]

              // 캐시에서 과거부터 현재까지의 전체 K, V를 가져옵니다.
              // [Shape] K_past: [2, N+1, 128]
              // [Shape] V_past: [2, N+1, 128]
              let k_past = &kv_cache[layer].k_cache;
              let v_past = &kv_cache[layer].v_cache;

              // ⑥ GQA (Grouped Query Attention) 연산
              // Qwen2는 Q 헤드 12개, KV 헤드 2개이므로, Q 헤드 6개당 1개의 KV 헤드를 공유합니다.
              // [Shape] attn_output_heads: [12, 1, 128]
              let mut attn_output_heads = Tensor3D::<NUM_Q_HEADS, 1, HEAD_DIM>::zeros();

              for q_idx in 0..NUM_Q_HEADS {
                  let kv_idx = q_idx / 6; // 0~5는 kv_idx 0, 6~11은 kv_idx 1 사용

                  // Q * K^T
                  // q[q_idx] 크기: [1, 128]
                  // k_past[kv_idx]^T 크기: [128, N+1]
                  // [Shape] scores: [1, N+1]
                  let mut scores = matmul(&q[q_idx], transpose(&k_past[kv_idx]));
                  scores = scale(&scores, 1.0 / sqrt(128.0));

                  // Softmax
                  // [Shape] probs: [1, N+1]
                  let probs = softmax(&scores);

                  // Probs * V
                  // probs 크기: [1, N+1]
                  // v_past[kv_idx] 크기: [N+1, 128]
                  // [Shape] head_out: [1, 128]
                  attn_output_heads[q_idx] = matmul(&probs, &v_past[kv_idx]);
              }

              // ⑦ Attention 출력 병합 및 O_proj
              // [Shape] attn_concat: [1, 1536] (12 * 128)
              let attn_concat = flatten_heads(&attn_output_heads);

              // [Shape] attn_out: [1, 1536]
              let attn_out = matmul(&attn_concat, &weights.layers[layer].o_proj);
        */
    }
}
