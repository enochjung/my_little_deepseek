use super::Error;

pub struct ByteLevelEngine;

// add_prefix_space=false, trim_offsets=false, use_regex=false.
impl ByteLevelEngine {
    pub fn new() -> Result<Self, Error> {
        Ok(Self)
    }

    pub fn pretokenize(&self, split_slices: &[&str]) -> Result<Vec<Vec<String>>, Error> {
        Ok(split_slices
            .iter()
            .map(|&slices| encode_token(slices))
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
            engine.pretokenize(&["Hello", "!"]).unwrap(),
            vec![tok("Hello"), tok("!")]
        );
    }

    #[test]
    fn pretokenize_case2_summarize() {
        let engine = ByteLevelEngine::new().unwrap();
        assert_eq!(
            engine
                .pretokenize(&[
                    "Summarize",
                    ":",
                    " Rust",
                    " ownership",
                    " prevents",
                    " data",
                    " races",
                    "."
                ])
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
                .pretokenize(&["What", " is", " ", "2", " +", " ", "2", "?"])
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
                .pretokenize(&[
                    "Whitespace",
                    " test",
                    ":",
                    " ",
                    " keep",
                    "  ",
                    " multiple",
                    " spaces",
                    ",",
                    " tabs",
                    "\t",
                    ",",
                    " and",
                    " blank",
                    " lines",
                    "\n\n",
                    "end",
                    ".",
                ])
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
                .pretokenize(&[
                    "Emoji",
                    " test",
                    ":",
                    " cats",
                    " 😺",
                    " rockets",
                    " 🚀",
                    " and",
                    " sparkles",
                    " ✨.",
                ])
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
                .pretokenize(&[
                    "Answer", " with", " JSON", ":", " {", " \"", "name", "\":", " \"", "Alice",
                    "\",", " \"", "age", "\":", " ", "2", "7", " }",
                ])
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
                .pretokenize(&[
                    "List", " these", ":", " ", "2", ",", " ", "3", ",", " ", "5", ",", " ", "7",
                    ",", " ", "1", "1", ",", " ", "1", "3", ",", " ", "1", "7", ",", " ", "1", "9",
                ])
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
                .pretokenize(&[
                    "Code",
                    " tip",
                    ":",
                    " avoid",
                    " unwrap",
                    "()",
                    " in",
                    " production",
                    " Rust",
                    "."
                ])
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
                .pretokenize(&[
                    "Compute", ":", " ", "1", "2", "7", " *", " ", "4", "3", " =", " ", "5", "4",
                    "6", "1",
                ])
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
                .pretokenize(&[
                    "Code", ":\n", "```", "python", "\n", "for", " i", " in", " range", "(", "3",
                    "):\n", "   ", " print", "(i", ")\n", "```",
                ])
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
