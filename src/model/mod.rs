mod attention;
mod embedding;
mod rms_normalizer;
mod tokenizer;

use crate::config::{Configure, Format};
use crate::session::{Ready, Session};
use crate::storage::*;
use crate::tensor::*;
use attention::Attention;
use embedding::Embedding;
use rms_normalizer::RMSNormalizer;
use std::collections::HashMap;
use std::ops::Range;
use tokenizer::Tokenizer;

pub struct Model {
    pub(crate) num_hidden_layers: usize,
    pub(crate) rms_norm_epsilon: f32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,

    tokenizer: Tokenizer,
    embedding: Embedding<F32, Host>,
    attentions: Vec<Attention<F32, Host>>,
}

impl Model {
    pub fn new(configure: Configure) -> Result<Self, crate::Error> {
        let num_hidden_layers = configure.num_hidden_layers;
        let rms_norm_epsilon = configure.rms_norm_epsilon;
        let hidden_size = configure.hidden_size;
        let intermediate_size = configure.intermediate_size;

        let unicode_format = &configure
            .unicode_format
            .ok_or(crate::Error::configure_failed("unicode format"))?;
        let composition_exclusion_format =
            &configure
                .composition_exclusion_format
                .ok_or(crate::Error::configure_failed(
                    "composition exclusion format",
                ))?;
        let merge_format = &configure
            .merge_format
            .ok_or(crate::Error::configure_failed("merge format"))?;
        let vocab_format = &configure
            .vocab_format
            .ok_or(crate::Error::configure_failed("vocab format"))?;

        let tokenizer = Tokenizer::new(
            &unicode_format,
            &composition_exclusion_format,
            &merge_format,
            &vocab_format,
        )?;

        let weight_format = &configure
            .weight_format
            .ok_or(crate::Error::configure_failed("weight format"))?;
        let (weight_storage, parse_weight) = weight_format.read()?;
        let weight_info = parse_weight(&weight_storage)
            .map(|weight| (weight.name.clone(), weight))
            .collect::<HashMap<_, _>>();
        let weight_info = ModelWeightInfo::new(num_hidden_layers, weight_info)?;

        let embedding = Embedding::new(&weight_storage, &weight_info.embed_tokens_weight)?;
        let attentions = weight_info
            .layers
            .iter()
            .map(|layer| Attention::new(&weight_storage, &layer.attention))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            num_hidden_layers,
            rms_norm_epsilon,
            hidden_size,
            intermediate_size,

            tokenizer,
            embedding,
            attentions,
        })
    }

    pub fn new_session(&self) -> Session<'_, Ready> {
        Session::<Ready>::new(self)
    }

    pub(crate) fn tokenize(&self, input: &str) -> Result<Vec<u32>, crate::Error> {
        self.tokenizer.tokenize(input)
    }

    pub(crate) fn build_embedding_vectors(
        &self,
        token_ids: &[u32],
    ) -> Result<Tensor<'static, Mut, F32, Host>, crate::Error> {
        let size = token_ids.len() * self.hidden_size as usize * F32::BYTES;
        let storage = MmapMut::new(size)?;
        let mut tensor =
            Tensor::<Mut, F32, Host>::new(storage, token_ids.len() as u32, self.hidden_size, true)?;

        for i in 0..token_ids.len() {
            let token_id = token_ids[i];
            let embed_tensor = self.embedding.word_embed(token_id)?;
            tensor.copy(i as u32, 0, &embed_tensor)?;
        }

        Ok(tensor)
    }

    pub(crate) fn apply_attention(
        &self,
        _target: &mut Tensor<Mut, F32, Host>,
        _tmp: &mut Host,
        _layer_idx: usize,
    ) -> Result<(), crate::Error> {
        todo!()
        /*
        let cloned = TensorMut::copy(tmp, 0, target)?;

        self.attentions[layer_idx].apply_attention(target, self.rms_norm_epsilon)?;
        target.residual(&cloned)?;

        Ok(())
        */
    }

    pub(crate) fn apply_feedforward(
        &self,
        _target: &mut Tensor<Mut, F32, Host>,
        _layer_idx: usize,
    ) -> Result<(), crate::Error> {
        todo!()

        /*
        // FeedForward(X: N*1536) -> N*1536
        {
            // post rms norm
            //// (model.layers.#.post_attention_layernorm.weight)

            // input (N*1536)

            // gate : Wgate x input (N*8960)
            //// (model.layers.#.mlp.gate_proj.weight)
            // up   : Wup   x input (N*8960)
            //// (model.layers.#.mlp.up_proj.weight)

            // gate_silu : SiLU(gate)
            //// SiLU : x / (1 + e^(-x)) (element op)

            // up_proj : up * gate_silu (element-wise) (N*8960)

            // down_proj : Wdown x up_proj (N*1536)
            //// (model.layers.#.mlp.down_proj.weight)
        }

        // residual (addition)
        // res := X + FeedForward(X)
        */
    }

    pub(crate) fn apply_lm_head(
        &self,
        _target: &mut Tensor<Mut, F32, Host>,
    ) -> Result<(), crate::Error> {
        todo!()

        // rms norm (model.norm.weight)
        // lm head (lm_head.weight)
        // append embedding if not finished
    }

    /*
    fn decode_one_step(&mut self, input_token: u32) -> u32 {
        // 1. Token Embedding
        // [Shape] input_x: [1, 1536]
        let mut x: Vector<HIDDEN_SIZE> = weights.embed_tokens[input_token];

        // 2. Transformer Layers 순회 (총 28개 층)
        for layer in 0..NUM_LAYERS {
            let residual = x.clone(); // Residual Connection을 위한 저장

            // ========================================================
            // [Attention Block]
            // ========================================================

            // ① Pre-RMS Norm
            // [Shape] norm_x: [1, 1536]
            let norm_x = rms_norm(&x, &weights.layers[layer].input_layernorm);

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

            // ⑧ 첫 번째 Residual Connection
            // [Shape] x: [1, 1536]
            x = add(&residual, &attn_out);

            // ========================================================
            // [MLP Block (SwiGLU)]
            // ========================================================
            let residual_mlp = x.clone();

            // ⑨ Post-Attention RMS Norm
            // [Shape] norm_x_mlp: [1, 1536]
            let norm_x_mlp = rms_norm(&x, &weights.layers[layer].post_attention_layernorm);

            // ⑩ Gate & Up Projection (Qwen2는 SwiGLU를 사용)
            // [Shape] gate_proj: [1, 8960]
            // [Shape] up_proj: [1, 8960]
            let gate_proj = matmul(&norm_x_mlp, &weights.layers[layer].gate_proj);
            let up_proj = matmul(&norm_x_mlp, &weights.layers[layer].up_proj);

            // ⑪ SiLU 활성화 함수 및 요소별 곱셈(Element-wise)
            // [Shape] activated: [1, 8960]
            let activated = elementwise_mul(&silu(&gate_proj), &up_proj);

            // ⑫ Down Projection
            // [Shape] mlp_out: [1, 1536]
            let mlp_out = matmul(&activated, &weights.layers[layer].down_proj);

            // ⑬ 두 번째 Residual Connection
            // [Shape] x: [1, 1536]
            x = add(&residual_mlp, &mlp_out);
        } // layer 루프 종료

        // ========================================================
        // [Final Output Block]
        // ========================================================

        // 3. Final RMS Norm
        // [Shape] final_x: [1, 1536]
        let final_x = rms_norm(&x, &weights.norm);

        // 4. LM Head (Vocabulary Projection)
        // tie_word_embeddings이 false이므로 독립적인 가중치 사용
        // [Shape] logits: [1, 151936]
        let logits = matmul(&final_x, &weights.lm_head);

        // 5. Sampling
        // logits에서 가장 확률이 높은 토큰(Argmax) 또는 샘플링을 통해 다음 토큰 결정
        let next_token = argmax(&logits);

        next_token
    }
    */
}

fn build_tensor_f32<'a>(
    storage_bf16: &Mmap,
    weight_info_bf16: &WeightInfo,
) -> Result<Tensor<'a, Own, F32, Host>, crate::Error> {
    let (nrow, ncol) = match weight_info_bf16.shape.as_slice() {
        [] => return Err(crate::Error::broken_data(0)),
        [ncol] => (1, *ncol),
        [nrow, ncol] => (*nrow, *ncol),
        _ => return Err(crate::Error::broken_data(0)),
    };
    let tensor_bf16 = Tensor::<Ref, BF16, Host>::new(
        storage_bf16,
        weight_info_bf16.offset.start,
        nrow,
        ncol,
        true,
    )?;

    let storage_f32 = MmapMut::new(nrow as usize * ncol as usize * F32::BYTES)?;
    let mut tensor_f32 = Tensor::<Mut, F32, Host>::new(storage_f32, nrow, ncol, true)?;
    tensor_f32.cast(&tensor_bf16)?;
    Ok(tensor_f32.to_readonly())
}

pub(crate) struct WeightInfo {
    pub(crate) name: String,
    pub(crate) shape: Vec<u32>,
    pub(crate) offset: Range<usize>,
}

struct AttentionLayerWeightInfo {
    input_layernorm_weight: WeightInfo,
    q_proj_bias: WeightInfo,
    q_proj_weight: WeightInfo,
    k_proj_bias: WeightInfo,
    k_proj_weight: WeightInfo,
    v_proj_bias: WeightInfo,
    v_proj_weight: WeightInfo,
    o_proj_weight: WeightInfo,
}

struct FeedforwardLayerWeightInfo {
    post_attention_layernorm_weight: WeightInfo,
    gate_proj_weight: WeightInfo,
    up_proj_weight: WeightInfo,
    down_proj_weight: WeightInfo,
}

struct LayerWeightInfo {
    attention: AttentionLayerWeightInfo,
    feedforward: FeedforwardLayerWeightInfo,
}

struct ModelWeightInfo {
    embed_tokens_weight: WeightInfo,
    layers: Vec<LayerWeightInfo>,
    norm_weight: WeightInfo,
    lm_head_weight: WeightInfo,
}

impl ModelWeightInfo {
    fn new(
        num_hidden_layers: usize,
        weight_info: HashMap<String, WeightInfo>,
    ) -> Result<Self, crate::Error> {
        let mut weight_info = weight_info;

        let embed_tokens_weight = take_tensor(&mut weight_info, "model.embed_tokens.weight")?;
        let norm_weight = take_tensor(&mut weight_info, "model.norm.weight")?;
        let lm_head_weight = take_tensor(&mut weight_info, "lm_head.weight")?;

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for layer_idx in 0..num_hidden_layers {
            let attention = AttentionLayerWeightInfo {
                input_layernorm_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.input_layernorm.weight"),
                )?,
                q_proj_bias: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                )?,
                q_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                )?,
                k_proj_bias: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                )?,
                k_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                )?,
                v_proj_bias: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                )?,
                v_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                )?,
                o_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                )?,
            };

            let feedforward = FeedforwardLayerWeightInfo {
                post_attention_layernorm_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                )?,
                gate_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                )?,
                up_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                )?,
                down_proj_weight: take_tensor(
                    &mut weight_info,
                    &format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                )?,
            };

            layers.push(LayerWeightInfo {
                attention,
                feedforward,
            });
        }

        if let Some((_, weight)) = weight_info.into_iter().next() {
            return Err(crate::Error::broken_data(0));
        }

        Ok(Self {
            embed_tokens_weight,
            layers,
            norm_weight,
            lm_head_weight,
        })
    }
}

fn take_tensor(
    tensors: &mut HashMap<String, WeightInfo>,
    key: &str,
) -> Result<WeightInfo, crate::Error> {
    tensors
        .remove(key)
        .ok_or_else(|| crate::Error::data_not_provided(key))
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::config::*;
    use crate::tensor::*;

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";
    const WEIGHT_PATH: &'static str = "model/model.safetensors";

    fn get_model() -> Model {
        let conf = Configure::new()
            .unicode_format(UnicodeFormat::UnicodeCharacterDatabase {
                path: UNICODE_PATH.to_string(),
            })
            .composition_exclusion_format(CompositionExclusionFormat::UnicodeCharacterDatabase {
                path: COMPOSITION_EXCLUSION_PATH.to_string(),
            })
            .merge_format(MergeFormat::HuggingFace {
                path: MERGE_PATH.to_string(),
            })
            .vocab_format(VocabFormat::HuggingFace {
                path: VOCAB_PATH.to_string(),
            })
            .weight_format(WeightFormat::Safetensor {
                path: WEIGHT_PATH.to_string(),
            });

        Model::new(conf).expect("initializing model should succeed")
    }

    fn assert_embedding_values<'t>(tensor: Tensor<'t, Ref, F32, Host>, expected_rows: &[[f32; 5]]) {
        let slice_tensor = tensor.slice(0..4, 0..5);
        let vec = expected_rows
            .iter()
            .flat_map(|row| row.iter())
            .map(|&val| val)
            .collect::<Vec<_>>();
        slice_tensor.assert(&vec);
    }

    #[test]
    fn case01_embedding_hello_world() {
        // Expected embedding values for "Hello, world!" -> token_ids: [9707, 11, 1879, 0]
        // Each row is the first 5 columns (0..5) of the embedding vector from test_model/embed.data
        let expected_rows = [
            // Row 0 (token_id: 9707)
            [
                0.02978515625,
                0.03662109375,
                0.0022430419921875,
                -0.001953125,
                0.05419921875,
            ],
            // Row 1 (token_id: 11)
            [
                0.04833984375,
                -0.031494140625,
                0.037353515625,
                0.0023193359375,
                -0.031982421875,
            ],
            // Row 2 (token_id: 1879)
            [
                0.026611328125,
                -0.062255859375,
                0.0054931640625,
                -0.0341796875,
                0.034912109375,
            ],
            // Row 3 (token_id: 0)
            [
                0.0732421875,
                0.03515625,
                -0.01422119140625,
                0.018798828125,
                -0.0230712890625,
            ],
        ];

        // Token IDs for "Hello, world!"
        let token_ids = [9707, 11, 1879, 0];

        let model = get_model();

        let embed_tensor = model
            .build_embedding_vectors(&token_ids)
            .expect("appending embedding vectors should succeed");

        let nrow = token_ids.len() as u32;
        let ncol = model.hidden_size;
        let embed_tensor_ref = embed_tensor.slice(0..nrow, 0..ncol);

        assert_embedding_values(embed_tensor_ref, &expected_rows);
    }
}
