mod normalizer;
mod pretokenizer;
mod tokenizer_model;

use crate::config::{CompositionExclusionFormat, MergeFormat, UnicodeFormat, VocabFormat};
use normalizer::Normalizer;
use pretokenizer::Pretokenizer;
use tokenizer_model::TokenizerModel;

pub(crate) struct Tokenizer {
    normalizer: Normalizer,
    pretokenizer: Pretokenizer,
    tokenizer_model: TokenizerModel,
}

impl Tokenizer {
    pub(crate) fn new(
        unicode_format: &UnicodeFormat,
        composition_exclusion_format: &CompositionExclusionFormat,
        merge_format: &MergeFormat,
        vocab_format: &VocabFormat,
    ) -> Result<Self, crate::Error> {
        let normalizer = Normalizer::new(unicode_format, composition_exclusion_format)?;
        let pretokenizer = Pretokenizer::new();
        let tokenizer_model = TokenizerModel::new(merge_format, vocab_format)?;

        Ok(Self {
            normalizer,
            pretokenizer,
            tokenizer_model,
        })
    }

    pub(crate) fn encode(&self, input: &str) -> Result<Vec<u32>, crate::Error> {
        let normalized_input = self.normalizer.execute(input)?;
        let pretokenized_input = self.pretokenizer.execute(&normalized_input);
        let tokens = self.tokenizer_model.encode(&pretokenized_input)?;

        Ok(tokens)
    }

    pub(crate) fn decode(&self, tokens: &[u32]) -> Result<(usize, String), crate::Error> {
        self.tokenizer_model.decode(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::Tokenizer;
    use crate::config::{CompositionExclusionFormat, MergeFormat, UnicodeFormat, VocabFormat};
    use std::sync::OnceLock;

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const COMPOSITION_EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";

    static PRECOMPUTED_TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

    fn get_tokenizer() -> &'static Tokenizer {
        PRECOMPUTED_TOKENIZER.get_or_init(|| {
            let unicode_format = UnicodeFormat::UnicodeCharacterDatabase {
                path: UNICODE_PATH.to_string(),
            };
            let composition_exclusion_format =
                CompositionExclusionFormat::UnicodeCharacterDatabase {
                    path: COMPOSITION_EXCLUSION_PATH.to_string(),
                };
            let merge_format = MergeFormat::HuggingFace {
                path: MERGE_PATH.to_string(),
            };
            let vocab_format = VocabFormat::HuggingFace {
                path: VOCAB_PATH.to_string(),
            };

            Tokenizer::new(
                &unicode_format,
                &composition_exclusion_format,
                &merge_format,
                &vocab_format,
            )
            .expect("initializing tokenizer should succeed")
        })
    }

    fn assert(input: &str, expected: &[u32]) {
        let tokenizer = get_tokenizer();

        let actual = tokenizer.encode(input).expect("tokenizing should succeed");
        assert_eq!(
            actual, expected,
            "expected: {:?}, actual: {:?}",
            expected, actual
        );
    }

    #[test]
    fn cafe_acute() {
        assert("Cafe\u{0301}", &[34, 2577, 963]);
    }

    #[test]
    fn chinese() {
        assert("中文分词测试", &[104811, 17177, 99689, 81705]);
    }

    #[test]
    fn hello_world() {
        assert("Hello, world!", &[9707, 11, 1879, 0]);
    }

    #[test]
    fn multi_spaces() {
        assert("a    b     c", &[64, 262, 293, 257, 272]);
    }

    #[test]
    fn multiline() {
        assert(
            "line1\nline2\nline3",
            &[1056, 16, 198, 1056, 17, 198, 1056, 18],
        );
    }

    #[test]
    fn tabs() {
        assert("tabs\tbetween\twords", &[30993, 2233, 10053, 197, 5761]);
    }

    #[test]
    fn leading_trailing_space() {
        assert(
            " leading and trailing spaces ",
            &[6388, 323, 27748, 12621, 220],
        );
    }

    #[test]
    fn json_snippet() {
        assert(
            "json = {\"a\": [1, 2, 3], \"ok\": true}",
            &[
                2236, 284, 5212, 64, 788, 508, 16, 11, 220, 17, 11, 220, 18, 1125, 330, 562, 788,
                830, 92,
            ],
        );
    }

    #[test]
    fn dialog01() {
        assert(
            "hi, can you help me debug this tokenizer?",
            &[6023, 11, 646, 498, 1492, 752, 7390, 419, 45958, 30],
        );
    }

    #[test]
    fn dialog02() {
        assert(
            "sure, what input is failing for you?",
            &[19098, 11, 1128, 1946, 374, 21394, 369, 498, 30],
        );
    }

    #[test]
    fn dialog03() {
        assert(
            "it breaks on multiple spaces, can you check?",
            &[275, 18303, 389, 5248, 12621, 11, 646, 498, 1779, 30],
        );
    }

    #[test]
    fn dialog04() {
        assert(
            "yes, send me the exact string please.",
            &[9693, 11, 3624, 752, 279, 4734, 914, 4486, 13],
        );
    }

    #[test]
    fn dialog05() {
        assert(
            "here: 'a    b     c' and it looks odd.",
            &[
                6739, 25, 364, 64, 262, 293, 257, 272, 6, 323, 432, 5868, 10322, 13,
            ],
        );
    }

    #[test]
    fn dialog06() {
        assert(
            "ok, I will compare token ids now.",
            &[562, 11, 358, 686, 9429, 3950, 14151, 1431, 13],
        );
    }

    #[test]
    fn dialog07() {
        assert(
            "quick check: does newline handling look right?",
            &[27763, 1779, 25, 1558, 39027, 11589, 1401, 1290, 30],
        );
    }

    #[test]
    fn dialog08() {
        assert(
            "I think so, but test line1\\nline2\\nline3 too.",
            &[
                40, 1744, 773, 11, 714, 1273, 1555, 16, 1699, 1056, 17, 1699, 1056, 18, 2238, 13,
            ],
        );
    }

    #[test]
    fn url() {
        assert(
            "please test url parsing: https://example.com/a?b=1",
            &[
                30021, 1273, 2515, 22314, 25, 3703, 1110, 8687, 905, 14186, 30, 65, 28, 16,
            ],
        );
    }

    #[test]
    fn numbers() {
        assert(
            "thanks, also verify numbers like -1 +2 3.14159.",
            &[
                45493, 11, 1083, 10146, 5109, 1075, 481, 16, 488, 17, 220, 18, 13, 16, 19, 16, 20,
                24, 13,
            ],
        );
    }
}
