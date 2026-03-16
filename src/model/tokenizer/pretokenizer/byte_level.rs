use crate::error::AppError;

pub struct ByteLevelEngine;

// add_prefix_space=false, trim_offsets=false, use_regex=false.
impl ByteLevelEngine {
    pub fn new() -> Result<Self, AppError> {
        Ok(Self)
    }

    pub fn pretokenize(
        &self,
        input: &str,
        split_spans: &[(usize, usize)],
    ) -> Result<Vec<Vec<String>>, AppError> {
        Ok(split_spans
            .iter()
            .map(|&(start, end)| encode_token(&input[start..end]))
            .collect())
    }
}

fn encode_token(token: &str) -> Vec<String> {
    let mut result = Vec::new();

    for &byte in token.as_bytes() {
        let mapped = byte_to_unicode(byte);
        result.push(mapped.to_string());
    }

    result
}

fn byte_to_unicode(byte: u8) -> char {
    let codepoint = match byte {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => byte as u32,
        0..=0x20 => 0x100 + byte as u32,
        0x7f..=0xa0 => 0x121 + (byte as u32 - 0x7f),
        0xad => 0x143,
    };

    char::from_u32(codepoint).expect("byte-level mapping should always be valid")
}

#[cfg(test)]
mod tests {
    use super::ByteLevelEngine;

    fn tok(s: &str) -> Vec<String> {
        s.chars().map(|c| c.to_string()).collect()
    }

    #[test]
    fn pretokenize_case1_hello() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine.pretokenize("Hello!", &[(0, 5), (5, 6)]).unwrap(),
            vec![tok("Hello"), tok("!")],
        );
    }

    #[test]
    fn pretokenize_case2_summarize() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Summarize: Rust ownership prevents data races.",
                    &[
                        (0, 9),
                        (9, 10),
                        (10, 15),
                        (15, 25),
                        (25, 34),
                        (34, 39),
                        (39, 45),
                        (45, 46),
                    ],
                )
                .unwrap(),
            vec![
                tok("Summarize"),
                tok(":"),
                tok("ĠRust"),
                tok("Ġownership"),
                tok("Ġprevents"),
                tok("Ġdata"),
                tok("Ġraces"),
                tok("."),
            ],
        );
    }

    #[test]
    fn pretokenize_case3_what_is_2_plus_2() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "What is 2 + 2?",
                    &[
                        (0, 4),
                        (4, 7),
                        (7, 8),
                        (8, 9),
                        (9, 11),
                        (11, 12),
                        (12, 13),
                        (13, 14),
                    ],
                )
                .unwrap(),
            vec![
                tok("What"),
                tok("Ġis"),
                tok("Ġ"),
                tok("2"),
                tok("Ġ+"),
                tok("Ġ"),
                tok("2"),
                tok("?"),
            ],
        );
    }

    #[test]
    fn pretokenize_case4_whitespace() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Whitespace test:  keep   multiple spaces, tabs\t, and blank lines\n\nend.",
                    &[
                        (0, 10),
                        (10, 15),
                        (15, 16),
                        (16, 17),
                        (17, 22),
                        (22, 24),
                        (24, 33),
                        (33, 40),
                        (40, 41),
                        (41, 46),
                        (46, 47),
                        (47, 48),
                        (48, 52),
                        (52, 58),
                        (58, 64),
                        (64, 66),
                        (66, 69),
                        (69, 70),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case5_emoji() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Emoji test: cats 😺 rockets 🚀 and sparkles ✨.",
                    &[
                        (0, 5),
                        (5, 10),
                        (10, 11),
                        (11, 16),
                        (16, 21),
                        (21, 29),
                        (29, 34),
                        (34, 38),
                        (38, 47),
                        (47, 52),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case6_json() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Answer with JSON: { \"name\": \"Alice\", \"age\": 27 }",
                    &[
                        (0, 6),
                        (6, 11),
                        (11, 16),
                        (16, 17),
                        (17, 19),
                        (19, 21),
                        (21, 25),
                        (25, 27),
                        (27, 29),
                        (29, 34),
                        (34, 36),
                        (36, 38),
                        (38, 41),
                        (41, 43),
                        (43, 44),
                        (44, 45),
                        (45, 46),
                        (46, 48),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case7_list_primes() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "List these: 2, 3, 5, 7, 11, 13, 17, 19",
                    &[
                        (0, 4),
                        (4, 10),
                        (10, 11),
                        (11, 12),
                        (12, 13),
                        (13, 14),
                        (14, 15),
                        (15, 16),
                        (16, 17),
                        (17, 18),
                        (18, 19),
                        (19, 20),
                        (20, 21),
                        (21, 22),
                        (22, 23),
                        (23, 24),
                        (24, 25),
                        (25, 26),
                        (26, 27),
                        (27, 28),
                        (28, 29),
                        (29, 30),
                        (30, 31),
                        (31, 32),
                        (32, 33),
                        (33, 34),
                        (34, 35),
                        (35, 36),
                        (36, 37),
                        (37, 38),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case8_code_tip() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Code tip: avoid unwrap() in production Rust.",
                    &[
                        (0, 4),
                        (4, 8),
                        (8, 9),
                        (9, 15),
                        (15, 22),
                        (22, 24),
                        (24, 27),
                        (27, 38),
                        (38, 43),
                        (43, 44),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case9_compute() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Compute: 127 * 43 = 5461",
                    &[
                        (0, 7),
                        (7, 8),
                        (8, 9),
                        (9, 10),
                        (10, 11),
                        (11, 12),
                        (12, 14),
                        (14, 15),
                        (15, 16),
                        (16, 17),
                        (17, 19),
                        (19, 20),
                        (20, 21),
                        (21, 22),
                        (22, 23),
                        (23, 24),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }

    #[test]
    fn pretokenize_case10_python_code() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(
                    "Code:\n```python\nfor i in range(3):\n    print(i)\n```",
                    &[
                        (0, 4),
                        (4, 6),
                        (6, 9),
                        (9, 15),
                        (15, 16),
                        (16, 19),
                        (19, 21),
                        (21, 24),
                        (24, 30),
                        (30, 31),
                        (31, 32),
                        (32, 35),
                        (35, 38),
                        (38, 44),
                        (44, 46),
                        (46, 48),
                        (48, 51),
                    ],
                )
                .unwrap(),
            vec![
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
        );
    }
}
