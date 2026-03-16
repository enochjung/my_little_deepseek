use std::fs::File;

use crate::error::AppError;

mod merges;
mod vocab;

use merges::MergesEngine;
use vocab::VocabEngine;

pub struct ModelEngine {
    #[allow(unused)]
    vocab_engine: VocabEngine,
    #[allow(unused)]
    merges_engine: MergesEngine,
}

impl ModelEngine {
    pub fn new(vocab_file: File, merges_file: File) -> Result<Self, AppError> {
        let vocab_engine = VocabEngine::new(vocab_file)?;
        let merges_engine = MergesEngine::new(merges_file)?;

        Ok(Self {
            vocab_engine,
            merges_engine,
        })
    }

    pub fn encode(&self, pretokenized: &[Vec<String>]) -> Result<Vec<u32>, AppError> {
        let mut token_ids = Vec::new();

        for word in pretokenized {
            let merged_word = self.merges_engine.merge(word)?;

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
    const VOCAB_PATH: &'static str = "model/vocab.json";
    const MERGES_PATH: &'static str = "model/merges.json";

    use std::fs::File;

    use super::ModelEngine;

    fn tok(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    fn build_engine() -> ModelEngine {
        let vocab_file = File::open(VOCAB_PATH).expect("failed to open vocab.json");
        let merges_file = File::open(MERGES_PATH).expect("failed to open merges.json");

        ModelEngine::new(vocab_file, merges_file).expect("failed to build ModelEngine")
    }

    #[test]
    fn encode_case1_hello() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[tok("Hello"), tok("!")])
                .expect("encode should succeed"),
            vec![9707, 0],
        );
    }

    #[test]
    fn encode_case2_summarize() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
                    tok("Summarize"),
                    tok(":"),
                    tok("ĠRust"),
                    tok("Ġownership"),
                    tok("Ġprevents"),
                    tok("Ġdata"),
                    tok("Ġraces"),
                    tok("."),
                ])
                .expect("encode should succeed"),
            vec![9190, 5612, 551, 25, 33789, 15278, 27934, 821, 20588, 13],
        );
    }

    #[test]
    fn encode_case3_what_is_2_plus_2() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
                    tok("What"),
                    tok("Ġis"),
                    tok("Ġ"),
                    tok("2"),
                    tok("Ġ+"),
                    tok("Ġ"),
                    tok("2"),
                    tok("?"),
                ])
                .expect("encode should succeed"),
            vec![3838, 374, 220, 17, 488, 220, 17, 30],
        );
    }

    #[test]
    fn encode_case4_whitespace() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                73804, 1273, 25, 220, 2506, 256, 5248, 12621, 11, 22398, 197, 11, 323, 10113, 5128,
                271, 408, 13,
            ],
        );
    }

    #[test]
    fn encode_case5_emoji() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                92731, 1273, 25, 19423, 26525, 118, 51998, 11162, 248, 222, 323, 15186, 642, 25521,
                101, 13,
            ],
        );
    }

    #[test]
    fn encode_case6_json() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                16141, 448, 4718, 25, 314, 330, 606, 788, 330, 61686, 497, 330, 424, 788, 220, 17,
                22, 335,
            ],
        );
    }

    #[test]
    fn encode_case7_list_primes() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                852, 1493, 25, 220, 17, 11, 220, 18, 11, 220, 20, 11, 220, 22, 11, 220, 16, 16, 11,
                220, 16, 18, 11, 220, 16, 22, 11, 220, 16, 24,
            ],
        );
    }

    #[test]
    fn encode_case8_code_tip() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![2078, 11552, 25, 5648, 79813, 368, 304, 5670, 33789, 13],
        );
    }

    #[test]
    fn encode_case9_compute() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                46254, 25, 220, 16, 17, 22, 353, 220, 19, 18, 284, 220, 20, 19, 21, 16
            ],
        );
    }

    #[test]
    fn encode_case10_python_code() {
        let engine = build_engine();
        assert_eq!(
            engine
                .encode(&[
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
                ])
                .expect("encode should succeed"),
            vec![
                2078, 510, 73594, 12669, 198, 1958, 600, 304, 2088, 7, 18, 982, 262, 1173, 1956,
                340, 73594,
            ],
        );
    }
}
