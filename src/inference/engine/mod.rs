mod embedding;
mod special_token;
mod tokenizer;

use super::{Error, ModelData, tensor};
use embedding::EmbeddingEngine;
use tokenizer::TokenizerEngine;

pub struct InferenceEngine<'a> {
    _model_data: &'a ModelData,
    tokens: Vec<u32>,
    tokenizer_engine: TokenizerEngine<'a>,
    embedding_engine: EmbeddingEngine<'a, tensor::F32>,
}

impl<'a> InferenceEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let tokens = Vec::new();
        let tokenizer_engine = TokenizerEngine::new(model_data)?;
        let embedding_engine = EmbeddingEngine::new(model_data)?;

        Ok(Self {
            _model_data: model_data,
            tokens,
            tokenizer_engine,
            embedding_engine,
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

        // TODO: Architect Tensor API.

        // do
        {
            // word embedding
            //// (model.embed_tokens.weight)
            let _embedded_tensor = self.embedding_engine.word_embed(self.tokens[0])?;

            // for each layer [0, 28)
            {
                // input rms norm
                //// (model.layers.#.input_layernorm.weight)

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

                // residual (addition)

                // FeedForward(X: 1536*N) -> 1536*N
                {
                    // post rms norm
                    //// (model.layers.#.post_attention_layernorm.weight)

                    // input (1536*N)

                    // gate : Wgate x input (8960*N)
                    //// (model.layers.#.mlp.gate_proj.weight)
                    // up   : Wup   x input (8960*N)
                    //// (model.layers.#.mlp.up_proj.weight)

                    // gate_silu : SiLU(gate)
                    //// SiLU : x / (1 + e^(-x)) (element op)

                    // up_proj : up * gate_silu (element-wise) (8960*N)

                    // down_proj : Wdown x up_proj (1536*N)
                    //// (model.layers.#.mlp.down_proj.weight)
                }

                // residual (addition)
                // res := X + FeedForward(X)
            }

            // rms norm (model.norm.weight)

            // lm head (lm_head.weight)
        }
        // until eos

        // return generated tokens with pretty format

        todo!()
    }
}
