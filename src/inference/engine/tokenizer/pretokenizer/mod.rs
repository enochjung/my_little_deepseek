mod byte_level;
mod split;

use super::Error;
use byte_level::ByteLevelEngine;
use split::SplitEngine;

pub struct PretokenizerEngine {
    split: SplitEngine,
    byte_level: ByteLevelEngine,
}

impl PretokenizerEngine {
    pub fn new() -> Result<Self, Error> {
        let split = SplitEngine::new()?;
        let byte_level = ByteLevelEngine::new()?;

        Ok(Self { split, byte_level })
    }

    pub fn pretokenize(&self, input: &str) -> Result<Vec<Vec<String>>, Error> {
        let split_slices = self.split.pretokenize(input)?;
        let byte_level_tokens = self.byte_level.pretokenize(&split_slices)?;

        Ok(byte_level_tokens)
    }
}
