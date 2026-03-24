mod merge;
mod vocab;

use super::{Error, ModelData};
use merge::MergeEngine;
use vocab::VocabEngine;

pub struct ModelEngine<'a> {
    #[allow(unused)]
    model_data: &'a ModelData,
    merge_engine: MergeEngine<'a>,
    vocab_engine: VocabEngine<'a>,
}

impl<'a> ModelEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let merge_engine = MergeEngine::new(model_data)?;
        let vocab_engine = VocabEngine::new(model_data)?;

        Ok(Self {
            model_data,
            merge_engine,
            vocab_engine,
        })
    }

    pub fn encode(&self, pretokenized: &[Vec<String>]) -> Result<Vec<u32>, Error> {
        let mut token_ids = Vec::new();

        for word in pretokenized {
            let merged_word = self.merge_engine.merge(word)?;

            for token in merged_word {
                let token_id = self.vocab_engine.tokenize(&token)?;
                token_ids.push(token_id);
            }
        }

        Ok(token_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MERGE_PATH: &'static str = "model/merges.json";
    const VOCAB_PATH: &'static str = "model/vocab.json";

    fn assert_encoding(input: &[Vec<String>], expected: &[u32]) {
        let model_data = ModelData::new("none.none", "none.none", MERGE_PATH, VOCAB_PATH)
            .expect("Error occured at initializing data");
        let model_engine =
            ModelEngine::new(&model_data).expect("Error occured at initializing tokenizer model");
        let encoded = model_engine.encode(input).expect("Error occured at encode");
        assert_eq!(
            encoded, expected,
            "Encoding of {:?} failed: encoded:{:?}, expected:{:?}",
            input, encoded, expected
        );
    }

    fn tok(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    #[test]
    fn encode_case1_hello() {
        assert_encoding(&[tok("Hello"), tok("!")], &vec![9707, 0]);
    }

    #[test]
    fn encode_case2_summarize() {
        assert_encoding(
            &[
                tok("Summarize"),
                tok(":"),
                tok("ĠRust"),
                tok("Ġownership"),
                tok("Ġprevents"),
                tok("Ġdata"),
                tok("Ġraces"),
                tok("."),
            ],
            &[9190, 5612, 551, 25, 33789, 15278, 27934, 821, 20588, 13],
        );
    }

    #[test]
    fn encode_case3_what_is_2_plus_2() {
        assert_encoding(
            &[
                tok("What"),
                tok("Ġis"),
                tok("Ġ"),
                tok("2"),
                tok("Ġ+"),
                tok("Ġ"),
                tok("2"),
                tok("?"),
            ],
            &[3838, 374, 220, 17, 488, 220, 17, 30],
        );
    }

    #[test]
    fn encode_case4_whitespace() {
        assert_encoding(
            &[
                tok("Whitespace"),
                tok("Ġtest"),
                tok(":"),
                tok("Ġ"),
                tok("Ġkeep"),
                tok("ĠĠ"),
                tok("Ġmultiple"),
                tok("Ġspaces"),
                tok(","),
                tok("Ġtabs"),
                tok("ĉ"),
                tok(","),
                tok("Ġand"),
                tok("Ġblank"),
                tok("Ġlines"),
                tok("ĊĊ"),
                tok("end"),
                tok("."),
            ],
            &[
                73804, 1273, 25, 220, 2506, 256, 5248, 12621, 11, 22398, 197, 11, 323, 10113, 5128,
                271, 408, 13,
            ],
        );
    }

    #[test]
    fn encode_case5_emoji() {
        assert_encoding(
            &[
                tok("Emoji"),
                tok("Ġtest"),
                tok(":"),
                tok("Ġcats"),
                tok("ĠðŁĺº"),
                tok("Ġrockets"),
                tok("ĠðŁļĢ"),
                tok("Ġand"),
                tok("Ġsparkles"),
                tok("Ġâľ¨."),
            ],
            &[
                92731, 1273, 25, 19423, 26525, 118, 51998, 11162, 248, 222, 323, 15186, 642, 25521,
                101, 13,
            ],
        );
    }

    #[test]
    fn encode_case6_json() {
        assert_encoding(
            &[
                tok("Answer"),
                tok("Ġwith"),
                tok("ĠJSON"),
                tok(":"),
                tok("Ġ{"),
                tok("Ġ\""),
                tok("name"),
                tok("\":"),
                tok("Ġ\""),
                tok("Alice"),
                tok("\","),
                tok("Ġ\""),
                tok("age"),
                tok("\":"),
                tok("Ġ"),
                tok("2"),
                tok("7"),
                tok("Ġ}"),
            ],
            &[
                16141, 448, 4718, 25, 314, 330, 606, 788, 330, 61686, 497, 330, 424, 788, 220, 17,
                22, 335,
            ],
        );
    }

    #[test]
    fn encode_case7_list_primes() {
        assert_encoding(
            &[
                tok("List"),
                tok("Ġthese"),
                tok(":"),
                tok("Ġ"),
                tok("2"),
                tok(","),
                tok("Ġ"),
                tok("3"),
                tok(","),
                tok("Ġ"),
                tok("5"),
                tok(","),
                tok("Ġ"),
                tok("7"),
                tok(","),
                tok("Ġ"),
                tok("1"),
                tok("1"),
                tok(","),
                tok("Ġ"),
                tok("1"),
                tok("3"),
                tok(","),
                tok("Ġ"),
                tok("1"),
                tok("7"),
                tok(","),
                tok("Ġ"),
                tok("1"),
                tok("9"),
            ],
            &[
                852, 1493, 25, 220, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 16, 16, 11,
                220, 16, 18, 11, 220, 16, 22, 11, 220, 16, 24,
            ],
        );
    }

    #[test]
    fn encode_case8_code_tip() {
        assert_encoding(
            &[
                tok("Code"),
                tok("Ġtip"),
                tok(":"),
                tok("Ġavoid"),
                tok("Ġunwrap"),
                tok("()"),
                tok("Ġin"),
                tok("Ġproduction"),
                tok("ĠRust"),
                tok("."),
            ],
            &[2078, 11552, 25, 5648, 79813, 368, 304, 5670, 33789, 13],
        );
    }

    #[test]
    fn encode_case9_compute() {
        assert_encoding(
            &[
                tok("Compute"),
                tok(":"),
                tok("Ġ"),
                tok("1"),
                tok("2"),
                tok("7"),
                tok("Ġ*"),
                tok("Ġ"),
                tok("4"),
                tok("3"),
                tok("Ġ="),
                tok("Ġ"),
                tok("5"),
                tok("4"),
                tok("6"),
                tok("1"),
            ],
            &[
                46254, 25, 220, 16, 17, 22, 353, 220, 19, 18, 284, 220, 20, 19, 21, 16,
            ],
        );
    }

    #[test]
    fn encode_case10_python_code() {
        assert_encoding(
            &[
                tok("Code"),
                tok(":Ċ"),
                tok("```"),
                tok("python"),
                tok("Ċ"),
                tok("for"),
                tok("Ġi"),
                tok("Ġin"),
                tok("Ġrange"),
                tok("("),
                tok("3"),
                tok("):Ċ"),
                tok("ĠĠĠ"),
                tok("Ġprint"),
                tok("(i"),
                tok(")Ċ"),
                tok("```"),
            ],
            &[
                2078, 510, 73594, 12669, 198, 1958, 600, 304, 2088, 7, 18, 982, 262, 1173, 1956,
                340, 73594,
            ],
        );
    }
}
