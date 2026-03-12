use std::fs::File;

use crate::error::AppError;
use crate::tokenizer::TokenizerEngine;

pub struct InferenceApp {
    previous_context: Option<String>,
    tokenizer_engine: TokenizerEngine,
}

impl InferenceApp {
    pub fn new(
        unicode_data_file: File,
        composition_exclusions_file: File,
    ) -> Result<Self, AppError> {
        let tokenizer_engine =
            TokenizerEngine::new(unicode_data_file, composition_exclusions_file)?;

        Ok(Self {
            previous_context: None,
            tokenizer_engine,
        })
    }

    #[allow(unused)]
    pub fn run_prompt(&mut self, user_input: &str) -> Result<String, AppError> {
        let mut context = self.previous_context.take();
        if context.is_none() {
            context = Some(String::new());
        }

        self.tokenizer_engine
            .apply_chat_template(context.as_mut().unwrap(), user_input)?;
        let tokens = self.tokenizer_engine.tokenize(context.as_ref().unwrap());

        // run transformer & decoder with tokens.
        // save generated context to previous_context.
        todo!()
    }
}
