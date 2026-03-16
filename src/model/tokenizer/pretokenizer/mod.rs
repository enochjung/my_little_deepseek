use crate::error::AppError;
mod byte_level;
mod split;

use byte_level::ByteLevelEngine;
use split::SplitEngine;

pub struct PretokenizerEngine {
    split: SplitEngine,
    byte_level: ByteLevelEngine,
}

impl PretokenizerEngine {
    pub fn new() -> Result<Self, AppError> {
        let split = SplitEngine::new()?;
        let byte_level = ByteLevelEngine::new()?;

        Ok(Self { split, byte_level })
    }

    pub fn pretokenize(&self, input: &str) -> Result<Vec<Vec<String>>, AppError> {
        let split_spans = self.split.pretokenize(input)?;
        let byte_level_tokens = self.byte_level.pretokenize(input, &split_spans)?;

        Ok(byte_level_tokens)
    }
}
