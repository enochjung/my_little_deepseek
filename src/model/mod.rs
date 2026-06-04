mod attention;
mod embedding;
mod rms_normalizer;
mod rope;
mod tokenizer;

use crate::config::{Configure, Format};
use crate::session::{KVCache, Session};
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
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rms_norm_epsilon: f32,
    pub(crate) hidden_size: u32,
    pub(crate) head_size: u32,
    pub(crate) intermediate_size: u32,
    pub(crate) rope_theta: f32,

    tokenizer: Tokenizer,
    embedding: Embedding<F32, Host>,
    attentions: Vec<Attention<F32, Host>>,
}

impl Model {
    pub fn new(configure: Configure) -> Result<Self, crate::Error> {
        let num_hidden_layers = configure.num_hidden_layers;
        let num_attention_heads = configure.num_attention_heads;
        let num_key_value_heads = configure.num_key_value_heads;
        let rms_norm_epsilon = configure.rms_norm_epsilon;
        let hidden_size = configure.hidden_size;
        let intermediate_size = configure.intermediate_size;
        let rope_theta = configure.rope_theta;

        if hidden_size % num_attention_heads as u32 != 0 {
            return Err(crate::Error::configure_failed(
                "hidden size & num_attention_heads mismatch",
            ));
        }
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(crate::Error::configure_failed(
                "num_attention_heads & num_key_value_heads mismatch",
            ));
        }
        let head_size = hidden_size / num_attention_heads as u32;

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
            num_attention_heads,
            num_key_value_heads,
            rms_norm_epsilon,
            hidden_size,
            intermediate_size,
            rope_theta,
            head_size,

            tokenizer,
            embedding,
            attentions,
        })
    }

    pub fn new_session(&self) -> Session<'_> {
        Session::new(self)
    }

    pub(crate) fn tokenize(&self, input: &str) -> Result<Vec<u32>, crate::Error> {
        self.tokenizer.tokenize(input)
    }

    pub(crate) fn prefill(
        &self,
        tokens: &[u32],
        kv_cache: &mut KVCache,
    ) -> Result<u32, crate::Error> {
        // TODO: impl real prefill

        let mut next_token = 0;
        for &token in tokens {
            next_token = self.decode(token, kv_cache)?;
        }
        Ok(next_token)
    }

    pub(crate) fn decode(&self, token: u32, kv_cache: &mut KVCache) -> Result<u32, crate::Error> {
        let tmp_storage = MmapMut::new(1 * self.hidden_size as usize * F32::BYTES)?;
        let mut residual = Tensor::<Mut, F32, Host>::new(tmp_storage, 1, self.hidden_size, true)?;

        let mut x = self.build_embedding_tensor(&[token])?;

        for layer in 0..self.num_hidden_layers {
            residual.copy(0, 0, &x)?;
            self.run_attention_block(&mut x, kv_cache, layer)?;
            x.add(&residual)?;

            residual.copy(0, 0, &x)?;
            self.run_mlp_block(&mut x, layer)?;
            x.add(&residual)?;
        }

        let next_token = self.run_output_block(x)?;
        Ok(next_token)
    }

    fn build_embedding_tensor(
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

    fn run_attention_block(
        &self,
        x: &mut Tensor<'static, Mut, F32, Host>,
        kv_cache: &mut KVCache,
        layer: usize,
    ) -> Result<(), crate::Error> {
        self.attentions[layer].run_attention(x, kv_cache, self.rms_norm_epsilon)
    }

    fn run_mlp_block(
        &self,
        x: &mut Tensor<'static, Mut, F32, Host>,
        layer: usize,
    ) -> Result<(), crate::Error> {
        todo!()
        /*

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
           //// SiLU : x / (1 + e^(-x)) (element op)

           // ⑫ Down Projection
           // [Shape] mlp_out: [1, 1536]
           let mlp_out = matmul(&activated, &weights.layers[layer].down_proj);
        */
    }

    fn run_output_block(&self, x: Tensor<'static, Mut, F32, Host>) -> Result<u32, crate::Error> {
        todo!()
        /*
        // [Shape] final_x: [1, 1536]
        let final_x = rms_norm(&x, &weights.norm);

        // 4. LM Head (Vocabulary Projection)
        // tie_word_embeddings이 false이므로 독립적인 가중치 사용
        // [Shape] logits: [1, 151936]
        let logits = matmul(&final_x, &weights.lm_head);

        // 5. Sampling
        // logits에서 가장 확률이 높은 토큰(Argmax) 또는 샘플링을 통해 다음 토큰 결정
        let next_token = argmax(&logits);

         */

        // rms norm (model.norm.weight)
        // lm head (lm_head.weight)
        // append embedding if not finished
    }
}

fn build_tensor_f32(
    storage_bf16: &Mmap,
    weight_info_bf16: &WeightInfo,
) -> Result<Tensor<'static, Own, F32, Host>, crate::Error> {
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
            .build_embedding_tensor(&token_ids)
            .expect("appending embedding vectors should succeed");

        let nrow = token_ids.len() as u32;
        let ncol = model.hidden_size;
        let embed_tensor_ref = embed_tensor.slice(0..nrow, 0..ncol);

        assert_embedding_values(embed_tensor_ref, &expected_rows);
    }
}
