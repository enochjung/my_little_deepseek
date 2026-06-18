#![feature(stdarch_x86_avx512_bf16)]

mod attention;
mod feed_forward;
mod kv_cache;
mod rms_norm;
mod rope;
mod sampling;
mod tensor;
mod token_embedding;
mod tokenizer;

pub use kv_cache::KVCache;

use backend_host::{Host, Mmap};
use config::{Configure, Format, WeightInfo};
use core::{Backend, BackendOps, ElemType, MLTError, MatrixLayout, MemoryMut, MemoryOwn};

use attention::{AttentionWeights, GroupedQueryAttention};
use feed_forward::FeedForward;
use sampling::Sampling;
use tensor::Tensor;
use token_embedding::TokenEmbedding;
use tokenizer::Tokenizer;

use std::arch::x86_64::bf16;
use std::collections::HashMap;

/// Represents the immutable, stateless neural network definition and its loaded weights.
///
/// Once initialized from a `Configure`, a `Model` remains constant and can be safely shared across
/// multiple inference sessions. It manages the allocation of all temporary buffers required for the
/// forward pass, supplying them to sub-modules as needed.
///
/// # Type Parameters
///
/// * `E`: The computational element type, determining the precision of tensor operations (e.g.,
///   `f32` or `f16`.
/// * `ED`: The Embedding Device. Specifies the physical memory location for tokenization and detokenization processes.
/// * `TD`: The Transformer Device. Specifies where the main transformer layer computations occur.
///
/// Conceptually, setting `ED` to `Cpu` and `TD` to `Gpu` (when implemented) allows the embedding layer
/// to operate on system memory while offloading heavy matrix multiplications to VRAM.
///
/// # Examples
///
/// ```no_run
/// use backend_host::Host;
///
/// let config = config::Configure::new();
/// // Configuration paths for vocab, merges, and weights would be set here.
/// let model = inference::Model::<f32, Host<f32>, Host<f32>>::new(config).unwrap();
/// ```
pub struct Model<T: ElemType, EB: BackendOps<T>, TB: BackendOps<T>> {
    pub head_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
    pub vocab_size: u32,

    tokenizer: Tokenizer,
    embedding: TokenEmbedding<T, EB::Operand>,
    attentions: Vec<GroupedQueryAttention<T, TB::Operand>>,
    feed_forwards: Vec<FeedForward<T, TB::Operand>>,
    sampling: Sampling<T, EB::Operand>,
}

impl Model<f32, Host<f32>, Host<f32>> {
    /// Initializes a new model architecture and maps weights into device memory.
    ///
    /// This function parses the provided configuration, constructs the tokenizer, and initializes
    /// the token embeddings, attention layers, feed-forward networks, and sampling layers.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`](MLTError) if the configuration is invalid, if required files
    /// (e.g., weights or vocabularies) cannot be read, or if there is a mismatch in architectural dimensions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// let config = config::Configure::new();
    /// let model = inference::Model::new(config).expect("Failed to load model weights and configuration");
    /// ```
    pub fn new(configure: Configure) -> Result<Self, MLTError> {
        let hidden_size = configure.hidden_size;
        let intermediate_size = configure.intermediate_size;
        let num_attention_heads = configure.num_attention_heads;
        let num_hidden_layers = configure.num_hidden_layers;
        let num_key_value_heads = configure.num_key_value_heads;
        let rms_norm_epsilon = configure.rms_norm_epsilon;
        let rope_theta = configure.rope_theta;
        let vocab_size = configure.vocab_size;

        if !hidden_size.is_multiple_of(num_attention_heads as u32) {
            return Err(MLTError::configure_failed(
                "hidden size & num_attention_heads mismatch",
            ));
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            return Err(MLTError::configure_failed(
                "num_attention_heads & num_key_value_heads mismatch",
            ));
        }
        let head_size = hidden_size / num_attention_heads as u32;

        let unicode_format = &configure
            .unicode_format
            .ok_or(MLTError::configure_failed("unicode format"))?;
        let composition_exclusion_format = &configure
            .composition_exclusion_format
            .ok_or(MLTError::configure_failed("composition exclusion format"))?;
        let merge_format = &configure
            .merge_format
            .ok_or(MLTError::configure_failed("merge format"))?;
        let vocab_format = &configure
            .vocab_format
            .ok_or(MLTError::configure_failed("vocab format"))?;

        let tokenizer = Tokenizer::new(
            unicode_format,
            composition_exclusion_format,
            merge_format,
            vocab_format,
        )?;

        let weight_format = &configure
            .weight_format
            .ok_or(MLTError::configure_failed("weight format"))?;
        let (weight_mem, parse_weight) = weight_format.read()?;
        let mut wi_map = parse_weight(&weight_mem)
            .map(|wi| (wi.name.clone(), wi))
            .collect();

        let embed = build_tensor(
            &weight_mem,
            take_info(&mut wi_map, "model.embed_tokens.weight")?,
        )?;
        let embedding = TokenEmbedding::new(embed, vocab_size, hidden_size);

        let attentions = (0..num_hidden_layers)
            .map(|li| {
                let norm = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.input_layernorm.weight"),
                    )?,
                )?;
                let q_bias = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.q_proj.bias"),
                    )?,
                )?;
                let q_weight = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.q_proj.weight"),
                    )?,
                )?;
                let k_bias = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.k_proj.bias"),
                    )?,
                )?;
                let k_weight = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.k_proj.weight"),
                    )?,
                )?;
                let v_bias = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.v_proj.bias"),
                    )?,
                )?;
                let v_weight = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.v_proj.weight"),
                    )?,
                )?;
                let o_weight = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.self_attn.o_proj.weight"),
                    )?,
                )?;
                let weights = AttentionWeights {
                    q_bias,
                    q_weight,
                    k_bias,
                    k_weight,
                    v_bias,
                    v_weight,
                    o_weight,
                };
                Ok(GroupedQueryAttention::new(
                    norm,
                    weights,
                    head_size,
                    num_attention_heads,
                    num_key_value_heads,
                    rms_norm_epsilon,
                    rope_theta,
                ))
            })
            .collect::<Result<_, _>>()?;

        let feed_forwards = (0..num_hidden_layers)
            .map(|li| {
                let norm = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.post_attention_layernorm.weight"),
                    )?,
                )?;
                let gate = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.mlp.gate_proj.weight"),
                    )?,
                )?;
                let up = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.mlp.up_proj.weight"),
                    )?,
                )?;
                let down = build_tensor(
                    &weight_mem,
                    take_info(
                        &mut wi_map,
                        &format!("model.layers.{li}.mlp.down_proj.weight"),
                    )?,
                )?;
                Ok(FeedForward::new(norm, gate, up, down, rms_norm_epsilon))
            })
            .collect::<Result<_, _>>()?;

        let sampling = Sampling::new(
            build_tensor(&weight_mem, take_info(&mut wi_map, "model.norm.weight")?)?,
            build_tensor(&weight_mem, take_info(&mut wi_map, "lm_head.weight")?)?,
            rms_norm_epsilon,
        );

        if !wi_map.is_empty() {
            return Err(MLTError::broken_data(0));
        }

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

impl<T: ElemType, EB: BackendOps<T>, TB: BackendOps<T>> Model<T, EB, TB> {
    pub fn tokenize(&self, input: &str) -> Result<Vec<u32>, MLTError> {
        self.tokenizer.encode(input)
    }

    pub fn decode(
        &self,
        kv_caches: &mut Vec<KVCache<T, TB::Operand>>,
        token_ids: &[u32],
    ) -> Result<u32, MLTError>
    where
        //<<EB::Operand as Memory<T>>::Base as MemoryOwn<T>>::Operator: BackendOps<T>,
        //<<TB::Operand as Memory<T>>::Base as MemoryOwn<T>>::Operator: BackendOps<T>,
        TB: Backend<T, Operand = EB::Operand>, // TODO
    {
        let h = self.hidden_size;
        let d = self.head_size;
        let t = token_ids.len() as u32;
        let nt = kv_caches[0].n() + t;
        let i = self.intermediate_size;
        let v = self.vocab_size;

        let new_tensor = |nrow: u32, ncol: u32| -> Result<Tensor<T, TB::Operand>, MLTError> {
            let mem = TB::Operand::new(nrow as usize * ncol as usize * size_of::<T>())?;
            let ml = MatrixLayout::new(0, nrow, ncol, ncol, 1);
            Tensor::new(mem, ml)
        };

        let mut tmp_2_x_d = new_tensor(2, d)?;
        let mut tmp_2t_x_h = new_tensor(2 * t, h)?;
        let mut tmp_t_x_nt = new_tensor(t, nt)?;
        let mut tmp_2t_x_i = new_tensor(2 * t, i)?;
        let mut tmp_1_x_v = new_tensor(1, v)?;
        let mut target_t_x_h = new_tensor(t, h)?;

        self.token_embedding(&mut target_t_x_h, token_ids)?;

        //todo!("move x from EB to TB");

        self.transformer(
            kv_caches,
            t,
            &mut target_t_x_h,
            &mut tmp_2_x_d,
            &mut tmp_2t_x_h,
            &mut tmp_t_x_nt,
            &mut tmp_2t_x_i,
        )?;
        let next_token =
            self.sampling(&mut target_t_x_h.slice_mut(t - 1..t, 0..h), &mut tmp_1_x_v)?;

        Ok(next_token)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> Result<(usize, String), MLTError> {
        self.tokenizer.decode(tokens)
    }

    fn token_embedding<D: MemoryMut<T, Base = EB::Operand>>(
        &self,
        target_t_x_h: &mut Tensor<T, D>,
        token_ids: &[u32],
    ) -> Result<(), MLTError> {
        let h = self.hidden_size;

        for i in 0..token_ids.len() as u32 {
            let embed = self.embedding.execute(token_ids[i as usize])?;
            target_t_x_h.slice_mut(i..i + 1, 0..h).copy(&embed)?;
        }

        Ok(())
    }

    fn transformer<
        D: MemoryMut<T, Base = TB::Operand>,
        T0: MemoryMut<T, Base = TB::Operand>,
        T1: MemoryMut<T, Base = TB::Operand>,
        T2: MemoryMut<T, Base = TB::Operand>,
        T3: MemoryMut<T, Base = TB::Operand>,
    >(
        &self,
        kv_caches: &mut [KVCache<T, TB::Operand>],
        t: u32,
        target_t_x_h: &mut Tensor<T, D>,
        tmp_2_x_d: &mut Tensor<T, T0>,
        tmp_2t_x_h: &mut Tensor<T, T1>,
        tmp_t_x_nt: &mut Tensor<T, T2>,
        tmp_2t_x_i: &mut Tensor<T, T3>,
    ) -> Result<(), MLTError> {
        let (mut residual_t_x_h, mut tmp_t_x_h) = tmp_2t_x_h.split_row(t)?;

        for layer in 0..self.num_hidden_layers {
            residual_t_x_h.copy(target_t_x_h)?;

            self.attentions[layer].execute(
                &mut kv_caches[layer],
                t,
                target_t_x_h,
                tmp_2_x_d,
                &mut tmp_t_x_h,
                tmp_t_x_nt,
            )?;

            target_t_x_h.add_assign(&residual_t_x_h)?;

            residual_t_x_h.copy(target_t_x_h)?;
            self.feed_forwards[layer].execute(t, target_t_x_h, tmp_2t_x_i)?;
            target_t_x_h.add_assign(&residual_t_x_h)?;
        }

        Ok(())
    }

    fn sampling<D: MemoryMut<T, Base = EB::Operand>, T0: MemoryMut<T, Base = EB::Operand>>(
        &self,
        target_1_x_h: &mut Tensor<T, D>,
        tmp_1_x_v: &mut Tensor<T, T0>,
    ) -> Result<u32, MLTError> {
        self.sampling.execute(target_1_x_h, tmp_1_x_v)
    }
}

fn take_info<T>(
    wi_map: &mut HashMap<String, config::WeightInfo<T>>,
    name: &str,
) -> Result<WeightInfo<T>, MLTError> {
    wi_map
        .remove(name)
        .ok_or_else(|| MLTError::data_not_provided(name))
}

fn build_tensor<T: ElemType>(
    src_mem: &Mmap<u8>,
    wi: WeightInfo<bf16>,
) -> Result<Tensor<T, Mmap<T>>, MLTError> {
    let (nrow, ncol) = match wi.shape.as_slice() {
        [] => return Err(MLTError::broken_data(0)),
        [ncol] => (1, *ncol),
        [nrow, ncol] => (*nrow, *ncol),
        _ => return Err(MLTError::broken_data(0)),
    };
    let len = nrow as usize * ncol as usize;
    let size = len * size_of::<T>();
    let mut dst_mem = Mmap::new(size)?;
    unsafe {
        let dst = dst_mem.as_mut_ptr();
        let src = src_mem.as_slice().as_ptr().byte_add(wi.offset.start);
        T::cast_bf16(dst, src, len);
    };
    let dst_ml = MatrixLayout::new(0, nrow, ncol, ncol, 1);
    Tensor::new(dst_mem, dst_ml)
}

#[cfg(test)]
mod tests {
    use super::Model;
    use crate::tensor::Tensor;
    use backend_host::{Host, Mmap};
    use config::{
        CompositionExclusionFormat, Configure, MergeFormat, UnicodeFormat, VocabFormat,
        WeightFormat,
    };
    use core::{MatrixLayout, MemoryOwn};

    const UNICODE_PATH: &'static str = "../../model/UnicodeData.txt";
    const COMPOSITION_EXCLUSION_PATH: &'static str = "../../model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "../../model/merges.json";
    const VOCAB_PATH: &'static str = "../../model/vocab.json";
    const WEIGHT_PATH: &'static str = "../../model/model.safetensors";

    fn get_model() -> Model<f32, Host<f32>, Host<f32>> {
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

        Model::new(conf).expect("Failed to initialize model")
    }

    #[test]
    fn embedding_hello_world() {
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
            Mmap::new(t as usize * h as usize * size_of::<f32>()).expect("Failed to allocate Mmap"),
            MatrixLayout::new(0, t, h, h, 1),
        )
        .expect("Failed to create tensor");

        model
            .token_embedding(&mut tensor, &token_ids)
            .expect("Failed to token embedding");

        tensor.slice(0..4, 0..5).assert(&expected_rows);
    }
}
