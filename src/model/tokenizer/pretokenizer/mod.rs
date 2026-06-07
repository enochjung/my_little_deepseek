mod byte_level;
mod split;

pub(crate) struct Pretokenizer;

impl Pretokenizer {
    pub(crate) fn new() -> Self {
        Self {}
    }

    pub(crate) fn execute(&self, input: &str) -> Vec<Vec<String>> {
        let split_slices = split::pretokenize(input);
        let byte_level_tokens = byte_level::pretokenize(&split_slices);

        byte_level_tokens
    }
}
