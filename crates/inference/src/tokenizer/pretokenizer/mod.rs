mod byte_level;
mod split;

use byte_level::ByteLevel;
use split::Split;

pub struct Pretokenizer {
    byte_level: ByteLevel,
    split: Split,
}

impl Pretokenizer {
    pub fn new() -> Self {
        let byte_level = ByteLevel::new();
        let split = Split::new();
        Self { byte_level, split }
    }

    pub fn execute(&self, input: &str) -> Vec<Vec<String>> {
        let split_slices = self.split.pretokenize(input);
        self.byte_level.pretokenize(&split_slices)
    }
}
