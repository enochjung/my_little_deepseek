use std::fs::File;

mod special_token;
mod tokenizer;

use crate::error::AppError;
use tokenizer::TokenizerEngine;

pub struct InferenceModelEngine {
    tokens: Vec<u32>,
    tokenizer_engine: TokenizerEngine,
}

impl InferenceModelEngine {
    pub fn new(
        unicode_data_file: File,
        composition_exclusions_file: File,
        vocab_file: File,
        merges_file: File,
    ) -> Result<Self, AppError> {
        let tokens = vec![special_token::BEGIN_OF_SENTENCE];
        let tokenizer_engine = TokenizerEngine::new(
            unicode_data_file,
            composition_exclusions_file,
            vocab_file,
            merges_file,
        )?;

        Ok(Self {
            tokens,
            tokenizer_engine,
        })
    }

    pub fn run_prompt(&mut self, user_input: &str) -> Result<String, AppError> {
        self.tokens.push(special_token::USER);

        let mut input_tokens = self.tokenizer_engine.tokenize(user_input)?;
        self.tokens.append(&mut input_tokens);

        self.tokens.push(special_token::ASSISTANT);
        self.tokens.push(special_token::THINK);

        // run transformer & decoder with tokens.

        // save generated context to previous_context.
        //// self.prompt = Prompt::from(generated_tokens);

        todo!()
    }
}
