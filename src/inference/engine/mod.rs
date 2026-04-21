mod embedding;
mod normalization;
mod special_token;
mod tokenizer;

use super::{Error, ModelData, tensor};
use embedding::EmbeddingEngine;
use normalization::NormalizationEngine;
use tensor::{F32, HostMemory, Tensor};
use tokenizer::TokenizerEngine;

const NUM_HIDDEN_LAYERS: usize = 28;

pub struct InferenceEngine<'a> {
    _model_data: &'a ModelData,
    tokens: Vec<u32>,
    tokenizer_engine: TokenizerEngine<'a>,
    embedding_engine: EmbeddingEngine<'a, tensor::F32>,
    normalization_engine: NormalizationEngine<'a, tensor::F32, tensor::HostMemory>,
}

impl<'a> InferenceEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let tokens = Vec::new();
        let tokenizer_engine = TokenizerEngine::new(model_data)?;
        let embedding_engine = EmbeddingEngine::new(model_data)?;
        let normalization_engine = NormalizationEngine::new(model_data)?;

        Ok(Self {
            _model_data: model_data,
            tokens,
            tokenizer_engine,
            embedding_engine,
            normalization_engine,
        })
    }

    pub fn run_prompt(&mut self, user_input: &str) -> Result<String, Error> {
        if self.tokens.is_empty() {
            self.tokens.push(special_token::BEGIN_OF_SENTENCE);
        }
        self.tokens.push(special_token::USER);

        let mut input_tokens = self.tokenizer_engine.tokenize(user_input)?;
        self.tokens.append(&mut input_tokens);

        self.tokens.push(special_token::ASSISTANT);
        self.tokens.push(special_token::THINK_START);

        let mut embedded_tensor =
            Tensor::<F32, HostMemory>::with_capacity(self.tokens.len() * 1536, 1536)?;

        // word embedding
        //// (model.embed_tokens.weight)
        for token_id in self.tokens.iter() {
            let tensor = self.embedding_engine.word_embed(*token_id)?;
            embedded_tensor.append(&tensor)?;
        }

        // do
        {
            // for each layer [0, 28)
            for layer_idx in 0..NUM_HIDDEN_LAYERS {
                // X

                // Attention(X: N*1536) -> N*1536
                {
                    // input = X
                    let mut attention_input = embedded_tensor.clone();

                    // input rms norm
                    //// (model.layers.#.input_layernorm.weight)
                    self.normalization_engine
                        .apply_input_rms_norm(layer_idx, &mut attention_input)?;

                    // q k
                    //// (model.layers.#.self_attn.q_proj.bias)
                    //// (model.layers.#.self_attn.k_proj.bias)
                    //// (model.layers.#.self_attn.q_proj.weight)
                    //// (model.layers.#.self_attn.k_proj.weight)

                    // rope(q, k)

                    // v
                    //// (model.layers.#.self_attn.v_proj.bias)
                    //// (model.layers.#.self_attn.v_proj.weight)

                    // concat header

                    // output projection
                    //// (model.layers.#.self_attn.o_proj.weight)
                }

                // residual (addition)
                // res := X + Attention(X)

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
            }

            // rms norm (model.norm.weight)

            // lm head (lm_head.weight)

            // append embedding if not finished
        }
        // until eos

        // return generated tokens with pretty format

        todo!()
    }
}
