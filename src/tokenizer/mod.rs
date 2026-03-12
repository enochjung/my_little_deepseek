use std::fs::File;

mod normalizer;
mod pretokenizer;

use crate::{error::AppError, tokenizer::pretokenizer::PretokenizerEngine};
use normalizer::NormalizerEngine;

pub struct TokenizerEngine {
    #[allow(unused)]
    normalizer_engine: NormalizerEngine,
    pretokenizer_engine: PretokenizerEngine,
}

impl TokenizerEngine {
    pub fn new(
        unicode_data_file: File,
        composition_exclusions_file: File,
    ) -> Result<Self, AppError> {
        let normalizer_engine =
            NormalizerEngine::new(unicode_data_file, composition_exclusions_file)?;
        let pretokenizer_engine = PretokenizerEngine::new()?;

        Ok(Self {
            normalizer_engine,
            pretokenizer_engine,
        })
    }

    pub fn apply_chat_template(
        &self,
        #[allow(unused)] context: &mut str,
        #[allow(unused)] user_input: &str,
    ) -> Result<(), AppError> {
        todo!()
    }

    pub fn tokenize(&self, #[allow(unused)] context: &str) -> Result<Vec<u32>, AppError> {
        // normalize
        let normalized_context = self.normalizer_engine.normalize(context)?;

        // pre-tokenizer
        //// split
        //// byte-level
        #[allow(unused)]
        let bytelevel_context = self.pretokenizer_engine.pretokenize(&normalized_context)?;

        // model
        // todo!()

        // post-processor
        // nop

        todo!()
    }
}
