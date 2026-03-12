use std::fs::File;

mod model;
mod normalizer;
mod pretokenizer;

use crate::error::AppError;
use model::ModelEngine;
use normalizer::NormalizerEngine;
use pretokenizer::PretokenizerEngine;

pub struct TokenizerEngine {
    normalizer_engine: NormalizerEngine,
    pretokenizer_engine: PretokenizerEngine,
    model_engine: ModelEngine,
}

impl TokenizerEngine {
    pub fn new(
        unicode_data_file: File,
        composition_exclusions_file: File,
        vocab_file: File,
        merges_file: File,
    ) -> Result<Self, AppError> {
        let normalizer_engine =
            NormalizerEngine::new(unicode_data_file, composition_exclusions_file)?;
        let pretokenizer_engine = PretokenizerEngine::new()?;
        let model_engine = ModelEngine::new(vocab_file, merges_file)?;

        Ok(Self {
            normalizer_engine,
            pretokenizer_engine,
            model_engine,
        })
    }

    pub fn tokenize(&self, input: &str) -> Result<Vec<u32>, AppError> {
        let normalized_input = self.normalizer_engine.normalize(input)?;
        let pretokenized_input = self.pretokenizer_engine.pretokenize(&normalized_input)?;
        let tokens = self.model_engine.encode(&pretokenized_input)?;

        Ok(tokens)
    }
}
