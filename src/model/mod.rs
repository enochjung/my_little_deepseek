mod attention;
mod embedding;
mod rms_normalizer;
mod tokenizer;

use crate::config::{Configure, Format};
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
    pub fn new<'a>(configure: Configure<'a>) -> Result<Self, crate::Error> {
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

    pub(crate) fn tokenize(&self, input: &str) -> Result<Vec<u32>, crate::Error> {
        self.tokenizer.tokenize(input)
    }

    pub(crate) fn append_embedding_vectors(
        &self,
        target: &mut TensorOwn<F32, Host>,
        token_ids: &[u32],
    ) -> Result<(), crate::Error> {
        for &token_id in token_ids {
            let embed_tensor = self.embedding.word_embed(token_id)?;
            target.append(&embed_tensor)?;
        }

        Ok(())
    }

    pub(crate) fn apply_attention(
        &self,
        target: &mut TensorOwn<F32, Host>,
        layer_idx: usize,
    ) -> Result<(), crate::Error> {
        let cloned = target.clone();

        self.attentions[layer_idx].apply_attention(target, self.rms_norm_epsilon)?;
        target.residual(&cloned)?;

        Ok(())
    }

    pub(crate) fn apply_feedforward(
        &self,
        target: &mut TensorOwn<F32, Host>,
        layer_idx: usize,
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
        target: &mut TensorOwn<F32, Host>,
    ) -> Result<(), crate::Error> {
        todo!()

        // rms norm (model.norm.weight)
        // lm head (lm_head.weight)
        // append embedding if not finished
    }
}

pub(crate) struct WeightInfo {
    pub(crate) name: String,
    pub(crate) shape: Vec<u32>,
    pub(crate) offset: Range<usize>,
}

impl<'a> TryFrom<(&'a Host, &WeightInfo)> for TensorRef<'a, BF16, Host> {
    type Error = crate::Error;

    fn try_from(value: (&'a Host, &WeightInfo)) -> Result<Self, Self::Error> {
        let (storage, weight_info) = value;

        let (nrow, ncol) = match weight_info.shape.as_slice() {
            [] => return Err(crate::Error::broken_data(&weight_info.name, 0)),
            [ncol] => (1, *ncol),
            [nrow, ncol] => (*nrow, *ncol),
            _ => return Err(crate::Error::broken_data(&weight_info.name, 0)),
        };

        TensorRef::new(storage, weight_info.offset.start, true, nrow, ncol, ncol)
    }
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
            return Err(crate::Error::broken_data(&weight.name, 0));
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
    use crate::storage::Host;
    use crate::tensor::*;

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";
    const WEIGHT_PATH: &'static str = "model/model.safetensors";

    fn assert_embedding_values(tensor: &TensorOwn<F32, Host>, expected_rows: &[[f32; 5]]) {
        let ptr = tensor.as_ptr();
        let shape = tensor.shape();
        let ncol = shape[1] as usize;

        unsafe {
            let f32_ptr = ptr as *const f32;
            for (row_idx, expected_row) in expected_rows.iter().enumerate() {
                for col_idx in 0..5 {
                    let offset = row_idx * ncol + col_idx;
                    let actual = *f32_ptr.add(offset);
                    let expected = expected_row[col_idx];
                    assert!(
                        (actual - expected).abs() < 0.0001,
                        "row {} col {} mismatch: actual {}, expected {}, diff {}",
                        row_idx,
                        col_idx,
                        actual,
                        expected,
                        (actual - expected).abs()
                    );
                }
            }
        }
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

        let conf = Configure::new()
            .unicode_format(UnicodeFormat::UnicodeCharacterDatabase { path: UNICODE_PATH })
            .composition_exclusion_format(CompositionExclusionFormat::UnicodeCharacterDatabase {
                path: COMPOSITION_EXCLUSION_PATH,
            })
            .merge_format(MergeFormat::HuggingFace { path: MERGE_PATH })
            .vocab_format(VocabFormat::HuggingFace { path: VOCAB_PATH })
            .weight_format(WeightFormat::Safetensor { path: WEIGHT_PATH });

        let model = Model::new(conf).expect("initializing model should succeed");

        let mut embed_tensor = TensorOwn::<F32, Host>::new(
            Host::new("embed_tensor", 4 * 1536 * 4).expect("creating Host should succeed"),
            true,
            0,
            1536,
        )
        .expect("creating embedding tensor should succeed");
        model
            .append_embedding_vectors(&mut embed_tensor, &token_ids)
            .expect("appending embedding vectors should succeed");

        assert_embedding_values(&embed_tensor, &expected_rows);
    }
}
