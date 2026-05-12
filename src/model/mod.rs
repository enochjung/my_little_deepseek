mod tokenizer;

use crate::config::Configure;
use crate::storage::*;
use crate::tensor::*;
use tokenizer::Tokenizer;

pub struct Model {
    num_hidden_layers: usize,
    rms_norm_epsilon: f32,

    tokenizer: Tokenizer,
    /*
    _model_data: &'a ModelData,
    tokens: Vec<u32>,
    embedding_engine: EmbeddingEngine<'a>,
    normalization_engine: NormalizationEngine<'a>,
     */
}

impl Model {
    pub fn new<'a>(configure: Configure<'a>) -> Result<Self, crate::Error> {
        let num_hidden_layers = configure.num_hidden_layers;
        let rms_norm_epsilon = configure.rms_norm_epsilon;

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

        // let embedding_engine = EmbeddingEngine::new(model_data)?;
        // let normalization_engine = NormalizationEngine::new(model_data)?;

        Ok(Self {
            num_hidden_layers,
            rms_norm_epsilon,
            tokenizer,
            // embedding_engine
            // normalization_engine
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
        todo!()
    }

    pub(crate) fn inference<T: Tensor<F32, Host>>(&self, input: T) -> Result<(), crate::Error> {
        todo!()
    }

    /*
    pub fn run_prompt(&mut self, user_input: &str) -> Result<String, Error> {
        if self.tokens.is_empty() {
            self.tokens.push(special_token::BEGIN_OF_SENTENCE);
        }
        self.tokens.push(special_token::USER);

        let mut input_tokens = self.tokenizer_engine.tokenize(user_input)?;
        self.tokens.append(&mut input_tokens);

        self.tokens.push(special_token::ASSISTANT);
        self.tokens.push(special_token::THINK_START);

        let mut embedded_tensor = Tensor::<F32>::with_capacity(self.tokens.len() * 1536, 1536)?;

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
    */
}
