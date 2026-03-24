mod model;
mod normalizer;
mod pretokenizer;

use super::{Error, ModelData};
use model::ModelEngine;
use normalizer::NormalizerEngine;
use pretokenizer::PretokenizerEngine;

pub struct TokenizerEngine<'a> {
    #[allow(unused)]
    model_data: &'a ModelData,
    normalizer_engine: NormalizerEngine<'a>,
    pretokenizer_engine: PretokenizerEngine,
    model_engine: ModelEngine<'a>,
}

impl<'a> TokenizerEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let normalizer_engine = NormalizerEngine::new(model_data)?;
        let pretokenizer_engine = PretokenizerEngine::new()?;
        let model_engine = ModelEngine::new(model_data)?;

        Ok(Self {
            model_data,
            normalizer_engine,
            pretokenizer_engine,
            model_engine,
        })
    }

    pub fn tokenize(&self, input: &str) -> Result<Vec<u32>, Error> {
        let normalized_input = self.normalizer_engine.normalize(input)?;
        let pretokenized_input = self.pretokenizer_engine.pretokenize(&normalized_input)?;
        let tokens = self.model_engine.encode(&pretokenized_input)?;

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelData;
    use super::TokenizerEngine;

    const UNICODE_PATH: &'static str = "model/UnicodeData.txt";
    const EXCLUSION_PATH: &'static str = "model/CompositionExclusions.txt";
    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";

    fn assert_token_ids(input: &str, expected: &[u32]) {
        let model_data = ModelData::new(UNICODE_PATH, EXCLUSION_PATH, MERGE_PATH, VOCAB_PATH)
            .expect("Error occured at initializing data");
        let tokenizer =
            TokenizerEngine::new(&model_data).expect("Error occured at initializing tokenizer");
        let actual = tokenizer
            .tokenize(input)
            .expect("Error occured at tokenizing");
        assert_eq!(actual, expected);
    }

    #[test]
    fn tokenizer_case1_cafe_acute() {
        assert_token_ids("Cafe\u{0301}", &[34, 2577, 963]);
    }

    #[test]
    fn tokenizer_case2_chinese() {
        assert_token_ids("中文分词测试", &[104811, 17177, 99689, 81705]);
    }

    #[test]
    fn tokenizer_case3_hello_world() {
        assert_token_ids("hello world", &[14990, 1879]);
    }

    #[test]
    fn tokenizer_case4_hello_world_upper() {
        assert_token_ids("HELLO WORLD", &[50712, 1593, 50891]);
    }

    #[test]
    fn tokenizer_case5_hello_world_punct() {
        assert_token_ids("Hello, world!", &[9707, 11, 1879, 0]);
    }

    #[test]
    fn tokenizer_case6_multi_spaces() {
        assert_token_ids("a    b     c", &[64, 262, 293, 257, 272]);
    }

    #[test]
    fn tokenizer_case7_multiline() {
        assert_token_ids(
            "line1\nline2\nline3",
            &[1056, 16, 198, 1056, 17, 198, 1056, 18],
        );
    }

    #[test]
    fn tokenizer_case8_tabs() {
        assert_token_ids("tabs\tbetween\twords", &[30993, 2233, 10053, 197, 5761]);
    }

    #[test]
    fn tokenizer_case9_leading_trailing_space() {
        assert_token_ids(
            " leading and trailing spaces ",
            &[6388, 323, 27748, 12621, 220],
        );
    }

    #[test]
    fn tokenizer_case10_json_snippet() {
        assert_token_ids(
            "json = {\"a\": [1, 2, 3], \"ok\": true}",
            &[
                2236, 284, 5212, 64, 788, 508, 16, 11, 220, 17, 11, 220, 18, 1125, 330, 562, 788,
                830, 92,
            ],
        );
    }

    #[test]
    fn tokenizer_case11_dialog_1() {
        assert_token_ids(
            "hi, can you help me debug this tokenizer?",
            &[6023, 11, 646, 498, 1492, 752, 7390, 419, 45958, 30],
        );
    }

    #[test]
    fn tokenizer_case12_dialog_2() {
        assert_token_ids(
            "sure, what input is failing for you?",
            &[19098, 11, 1128, 1946, 374, 21394, 369, 498, 30],
        );
    }

    #[test]
    fn tokenizer_case13_dialog_3() {
        assert_token_ids(
            "it breaks on multiple spaces, can you check?",
            &[275, 18303, 389, 5248, 12621, 11, 646, 498, 1779, 30],
        );
    }

    #[test]
    fn tokenizer_case14_dialog_4() {
        assert_token_ids(
            "yes, send me the exact string please.",
            &[9693, 11, 3624, 752, 279, 4734, 914, 4486, 13],
        );
    }

    #[test]
    fn tokenizer_case15_dialog_5() {
        assert_token_ids(
            "here: 'a    b     c' and it looks odd.",
            &[
                6739, 25, 364, 64, 262, 293, 257, 272, 6, 323, 432, 5868, 10322, 13,
            ],
        );
    }

    #[test]
    fn tokenizer_case16_dialog_6() {
        assert_token_ids(
            "ok, I will compare token ids now.",
            &[562, 11, 358, 686, 9429, 3950, 14151, 1431, 13],
        );
    }

    #[test]
    fn tokenizer_case17_dialog_7() {
        assert_token_ids(
            "quick check: does newline handling look right?",
            &[27763, 1779, 25, 1558, 39027, 11589, 1401, 1290, 30],
        );
    }

    #[test]
    fn tokenizer_case18_dialog_8() {
        assert_token_ids(
            "I think so, but test line1\\nline2\\nline3 too.",
            &[
                40, 1744, 773, 11, 714, 1273, 1555, 16, 1699, 1056, 17, 1699, 1056, 18, 2238, 13,
            ],
        );
    }

    #[test]
    fn tokenizer_case19_url() {
        assert_token_ids(
            "please test url parsing: https://example.com/a?b=1",
            &[
                30021, 1273, 2515, 22314, 25, 3703, 1110, 8687, 905, 14186, 30, 65, 28, 16,
            ],
        );
    }

    #[test]
    fn tokenizer_case20_numbers() {
        assert_token_ids(
            "thanks, also verify numbers like -1 +2 3.14159.",
            &[
                45493, 11, 1083, 10146, 5109, 1075, 481, 16, 488, 17, 220, 18, 13, 16, 19, 16, 20,
                24, 13,
            ],
        );
    }
}
