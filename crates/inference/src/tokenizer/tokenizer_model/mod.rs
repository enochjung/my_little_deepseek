mod merge;
mod vocab;

use config::{MergeFormat, VocabFormat};
use core::MLTError;
use merge::Merge;
use vocab::Vocab;

pub struct TokenizerModel {
    merge: Merge,
    vocab: Vocab,
}

impl TokenizerModel {
    pub fn new(merge_format: &MergeFormat, vocab_format: &VocabFormat) -> Result<Self, MLTError> {
        let merge = Merge::new(merge_format)?;
        let vocab = Vocab::new(vocab_format)?;

        Ok(Self { merge, vocab })
    }

    pub fn encode(&self, pretokenized: &[Vec<String>]) -> Result<Vec<u32>, MLTError> {
        let mut token_ids = Vec::new();

        for word in pretokenized {
            let merged_word = self.merge.execute(word)?;

            for token in merged_word {
                let token_id = self.vocab.encode(&token)?;
                token_ids.push(token_id);
            }
        }

        Ok(token_ids)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<(usize, String), MLTError> {
        self.vocab.decode(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::TokenizerModel;
    use config::{MergeFormat, VocabFormat};

    const MERGE_PATH: &'static str = "../../model/merges.json";
    const VOCAB_PATH: &'static str = "../../model/vocab.json";

    fn assert(input: &[Vec<String>], expected: &[u32]) {
        let merge_format = MergeFormat::HuggingFace {
            path: MERGE_PATH.to_string(),
        };
        let vocab_format = VocabFormat::HuggingFace {
            path: VOCAB_PATH.to_string(),
        };
        let model_engine = TokenizerModel::new(&merge_format, &vocab_format)
            .expect("Failed to initialize tokenizer model");
        let actual = model_engine.encode(input).expect("Failed to encode text");
        assert_eq!(
            actual, expected,
            "expected:{:?}, actual:{:?}",
            expected, actual
        );
    }

    fn tok(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    #[test]
    fn hello() {
        assert(&[tok("Hello"), tok("!")], &[9707, 0]);
    }

    #[test]
    fn summarize() {
        assert(
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
    fn math_question() {
        assert(
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
    fn whitespace() {
        assert(
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
    fn emoji() {
        assert(
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
    fn json() {
        assert(
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
    fn list_numbers() {
        assert(
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
    fn code_tip() {
        assert(
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
    fn compute() {
        assert(
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
    fn python_code() {
        assert(
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
