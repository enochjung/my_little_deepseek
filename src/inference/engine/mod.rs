mod special_token;
mod tokenizer;

use super::{Error, ModelData};
use tokenizer::TokenizerEngine;

pub struct InferenceEngine<'a> {
    #[allow(unused)]
    model_data: &'a ModelData,
    tokens: Vec<u32>,
    tokenizer_engine: TokenizerEngine<'a>,
}

impl<'a> InferenceEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let tokens = Vec::new();
        let tokenizer_engine = TokenizerEngine::new(model_data)?;

        Ok(Self {
            model_data,
            tokens,
            tokenizer_engine,
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

        // do
        // - run decoder with tokens
        // - select token
        // until eos

        // return generated tokens with pretty format

        todo!()
    }
}
