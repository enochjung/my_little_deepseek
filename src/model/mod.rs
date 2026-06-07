mod attention;
mod feed_forward;
mod rms_norm;
mod rope;
mod sampling;
mod token_embedding;
mod tokenizer;

use crate::config::{Configure, Format};
use crate::device::{Cpu, Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::session::{KVCache, Session};
use crate::tensor::{ElemType, F32, Tensor};
use attention::Attention;
use feed_forward::FeedForward;
use rms_norm::RMSNorm;
use rope::RoPE;
use sampling::Sampling;
use std::collections::HashMap;
use std::ops::Range;
use token_embedding::TokenEmbedding;
use tokenizer::Tokenizer;

#[allow(private_bounds)]
pub struct Model<E: ElemType, ED: Device, TD: Device> {
    pub(crate) head_size: u32,
    pub(crate) hidden_size: u32,
    pub(crate) intermediate_size: u32,
    #[allow(dead_code)]
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    #[allow(dead_code)]
    pub(crate) rms_norm_epsilon: f32,
    #[allow(dead_code)]
    pub(crate) rope_theta: f32,
    pub(crate) vocab_size: u32,

    tokenizer: Tokenizer,
    embedding: TokenEmbedding<E, ED>,
    attentions: Vec<Attention<E, TD>>,
    feed_forwards: Vec<FeedForward<E, TD>>,
    sampling: Sampling<E, ED>,
}

impl Model<F32, Cpu, Cpu> {
    pub fn new(configure: Configure) -> Result<Self, crate::Error> {
        let hidden_size = configure.hidden_size;
        let intermediate_size = configure.intermediate_size;
        let num_attention_heads = configure.num_attention_heads;
        let num_hidden_layers = configure.num_hidden_layers;
        let num_key_value_heads = configure.num_key_value_heads;
        let rms_norm_epsilon = configure.rms_norm_epsilon;
        let rope_theta = configure.rope_theta;
        let vocab_size = configure.vocab_size;

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

        let embedding = TokenEmbedding::new(&weight_storage, &weight_info.embed_tokens_weight)?;
        let attentions = weight_info
            .layers
            .iter()
            .map(|layer| {
                Attention::new(
                    &weight_storage,
                    &layer.attention,
                    head_size,
                    num_attention_heads,
                    num_key_value_heads,
                    rms_norm_epsilon,
                    rope_theta,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let feed_forwards = weight_info
            .layers
            .iter()
            .map(|layer| {
                FeedForward::new(
                    &weight_storage,
                    &layer.feed_forward,
                    intermediate_size,
                    rms_norm_epsilon,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sampling = Sampling::new(
            &weight_storage,
            &weight_info.norm_weight,
            &weight_info.lm_head_weight,
            rms_norm_epsilon,
        )?;

        Ok(Self {
            head_size,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            num_key_value_heads,
            rms_norm_epsilon,
            rope_theta,
            vocab_size,

            tokenizer,
            embedding,
            attentions,
            feed_forwards,
            sampling,
        })
    }
}

#[allow(private_bounds)]
impl<E: ElemType, ED: Device, TD: Device> Model<E, ED, TD> {
    pub fn new_session(&self) -> Result<Session<'_, E, ED, TD>, crate::Error> {
        Session::new(self)
    }

    pub(crate) fn tokenize(&self, input: &str) -> Result<Vec<u32>, crate::Error> {
        self.tokenizer.encode(input)
    }
}

#[allow(private_bounds)]
impl<E: ElemType, ED: Device, TD: Device> Model<E, ED, TD>
where
    ED::Base: DeviceOps<E>,
    TD::Base: DeviceOps<E>,
{
    pub(crate) fn prefill<OD: OwnedDevice>(
        &self,
        kv_caches: &mut Vec<KVCache<E, OD>>,
        tokens: &[u32],
    ) -> Result<u32, crate::Error>
    where
        TD: Device<Base = OD>,
        ED: Device<Base = OD>, // temporary
    {
        //todo!("impl real prefill");

        let mut next_token = 0;

        for &token in tokens {
            next_token = self.decode(kv_caches, token)?;
        }
        Ok(next_token)
    }

    pub(crate) fn decode<OD: OwnedDevice>(
        &self,
        kv_caches: &mut Vec<KVCache<E, OD>>,
        token: u32,
    ) -> Result<u32, crate::Error>
    where
        TD: Device<Base = OD>,
        ED: Device<Base = OD>, // temporary
    {
        let h = self.hidden_size;
        let d = self.head_size;
        let n1 = kv_caches[0].n() + 1;
        let i = self.intermediate_size;
        let v = self.vocab_size;

        let mut tmp_2_x_d = Tensor::new(TD::Base::new(2 * d as usize * E::BYTES)?, 0, 2, d, d)?;
        let mut tmp_2_x_h = Tensor::new(TD::Base::new(2 * h as usize * E::BYTES)?, 0, 2, h, h)?;
        let mut tmp_1_x_n1 = Tensor::new(TD::Base::new(1 * n1 as usize * E::BYTES)?, 0, 1, n1, n1)?;
        let mut tmp_3_x_i = Tensor::new(TD::Base::new(3 * i as usize * E::BYTES)?, 0, 3, i, i)?;
        let mut tmp_1_x_v = Tensor::new(TD::Base::new(1 * v as usize * E::BYTES)?, 0, 1, v, v)?;

        let mut x = Tensor::new(ED::Base::new(1 * h as usize * E::BYTES)?, 0, 1, h, h)?;

        self.token_embedding(&mut x, &[token])?;

        //todo!("move x from ED to TD");

        self.transformer(
            kv_caches,
            &mut x,
            &mut tmp_2_x_d,
            &mut tmp_2_x_h,
            &mut tmp_1_x_n1,
            &mut tmp_3_x_i,
        )?;
        let next_token = self.sampling(x, &mut tmp_1_x_v)?;

        Ok(next_token)
    }

    pub(crate) fn detokenize(&self, tokens: &[u32]) -> Result<(usize, String), crate::Error> {
        self.tokenizer.decode(tokens)
    }

    fn token_embedding<M0: MutableDevice<Base = ED::Base>>(
        &self,
        target_t_x_h: &mut Tensor<E, M0>,
        token_ids: &[u32],
    ) -> Result<(), crate::Error> {
        let h = self.hidden_size;

        for i in 0..token_ids.len() as u32 {
            let embed = self.embedding.execute(token_ids[i as usize])?;
            target_t_x_h.slice_mut(i..i + 1, 0..h).copy(&embed)?;
        }

        Ok(())
    }

    fn transformer<
        OD: OwnedDevice,
        M0: MutableDevice<Base = TD::Base>,
        M1: MutableDevice<Base = TD::Base>,
        M2: MutableDevice<Base = TD::Base>,
        M3: MutableDevice<Base = TD::Base>,
        M4: MutableDevice<Base = TD::Base>,
    >(
        &self,
        kv_caches: &mut Vec<KVCache<E, OD>>,
        x: &mut Tensor<E, M0>,
        tmp_2_x_d: &mut Tensor<E, M1>,
        tmp_2_x_h: &mut Tensor<E, M2>,
        tmp_1_x_n1: &mut Tensor<E, M3>,
        tmp_3_x_i: &mut Tensor<E, M4>,
    ) -> Result<(), crate::Error>
    where
        TD: Device<Base = OD>,
    {
        let h = self.hidden_size;
        let tmp_2_x_h = tmp_2_x_h.slice_mut(0..2, 0..h);
        let (mut residual, mut tmp_1_x_h) = tmp_2_x_h.split_row(1)?;

        for layer in 0..self.num_hidden_layers {
            residual.copy(&x)?;

            self.attentions[layer].execute(
                &mut kv_caches[layer],
                x,
                tmp_2_x_d,
                &mut tmp_1_x_h,
                tmp_1_x_n1,
            )?;

            x.add(&residual)?;

            residual.copy(&x)?;
            self.feed_forwards[layer].execute(x, tmp_3_x_i)?;
            x.add(&residual)?;
        }

        Ok(())
    }

    fn sampling<M0: MutableDevice<Base = ED::Base>, M1: MutableDevice<Base = ED::Base>>(
        &self,
        x: Tensor<E, M0>,
        tmp_1_x_v: &mut Tensor<E, M1>,
    ) -> Result<u32, crate::Error> {
        self.sampling.execute(x, tmp_1_x_v)
    }
}

fn build_casted_tensor<E: ElemType, OD: OwnedDevice + DeviceOps<E>>(
    storage_bf16: &OD,
    weight_info_bf16: &WeightInfo,
) -> Result<Tensor<E, OD>, crate::Error> {
    let (nrow, ncol) = match weight_info_bf16.shape.as_slice() {
        [] => return Err(crate::Error::broken_data(0)),
        [ncol] => (1, *ncol),
        [nrow, ncol] => (*nrow, *ncol),
        _ => return Err(crate::Error::broken_data(0)),
    };
    let tensor_bf16 = Tensor::new(
        &*storage_bf16,
        weight_info_bf16.offset.start,
        nrow,
        ncol,
        ncol,
    )?;

    let mut tensor_f32 = Tensor::new(
        OD::new(nrow as usize * ncol as usize * F32::BYTES)?,
        0,
        nrow,
        ncol,
        ncol,
    )?;
    tensor_f32.cast_from_bf16(&tensor_bf16)?;
    Ok(tensor_f32)
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

struct FeedForwardLayerWeightInfo {
    post_attention_layernorm_weight: WeightInfo,
    gate_proj_weight: WeightInfo,
    up_proj_weight: WeightInfo,
    down_proj_weight: WeightInfo,
}

struct LayerWeightInfo {
    attention: AttentionLayerWeightInfo,
    feed_forward: FeedForwardLayerWeightInfo,
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

            let feed_forward = FeedForwardLayerWeightInfo {
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
                feed_forward,
            });
        }

        if weight_info.into_iter().next().is_some() {
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
    use crate::config::{
        CompositionExclusionFormat, Configure, MergeFormat, UnicodeFormat, VocabFormat,
        WeightFormat,
    };
    use crate::device::{Cpu, OwnedDevice};
    use crate::tensor::{ElemType, F32, Tensor};

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";
    const WEIGHT_PATH: &'static str = "model/model.safetensors";

    fn get_model() -> Model<F32, Cpu, Cpu> {
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

        Model::new(conf).expect("`Model::new` should succeed")
    }

    #[test]
    fn case01_embedding_hello_world() {
        // "Hello, world!" -> token_ids: [9707, 11, 1879, 0]
        let token_ids = [9707, 11, 1879, 0];
        let expected_rows = [
            [
                0.02978515625,
                0.03662109375,
                0.0022430419921875,
                -0.001953125,
                0.05419921875,
            ],
            [
                0.04833984375,
                -0.031494140625,
                0.037353515625,
                0.0023193359375,
                -0.031982421875,
            ],
            [
                0.026611328125,
                -0.062255859375,
                0.0054931640625,
                -0.0341796875,
                0.034912109375,
            ],
            [
                0.0732421875,
                0.03515625,
                -0.01422119140625,
                0.018798828125,
                -0.0230712890625,
            ],
        ];

        let model = get_model();
        let t = token_ids.len() as u32;
        let h = model.hidden_size;

        let mut tensor = Tensor::new(
            Cpu::new((t * h) as usize * F32::BYTES).expect("`Cpu::new` should succeed"),
            0,
            t,
            h,
            h,
        )
        .expect("`Tensor::new` should succeed");

        model
            .token_embedding(&mut tensor, &token_ids)
            .expect("`Model::token_embedding` should succeed");

        tensor.slice(0..4, 0..5).assert(&expected_rows);
    }
}
