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

    fn assert(input: &[&str], expected: &[Vec<String>]) {
        let engine = ByteLevelEngine::new().expect("initializing byte-level should succeed");
        let actual = engine
            .pretokenize(input)
            .expect("byte-level pretokenization should succeed");
        assert_eq!(
            actual, expected,
            "actual: {:?}, expected: {:?}",
            actual, expected
        );
    }

    #[test]
    fn case01_hello() {
        assert(&["Hello", "!"], &[tok("Hello"), tok("!")]);
    }

    #[test]
    fn case02_summarize() {
        assert(
            &[
                "Summarize",
                ":",
                " Rust",
                " ownership",
                " prevents",
                " data",
                " races",
                ".",
            ],
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
        );
    }

    #[test]
    fn case03_what_is_2_plus_2() {
        assert(
            &["What", " is", " ", "2", " +", " ", "2", "?"],
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
        );
    }

    #[test]
    fn case04_whitespace() {
        assert(
            &[
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
            ],
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
        );
    }

    #[test]
    fn case05_emoji() {
        assert(
            &[
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
            ],
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
        );
    }

    #[test]
    fn case06_json() {
        assert(
            &[
                "Answer", " with", " JSON", ":", " {", " \"", "name", "\":", " \"", "Alice", "\",",
                " \"", "age", "\":", " ", "2", "7", " }",
            ],
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
        );
    }

    #[test]
    fn case07_list_primes() {
        assert(
            &[
                "List", " these", ":", " ", "2", ",", " ", "3", ",", " ", "5", ",", " ", "7", ",",
                " ", "1", "1", ",", " ", "1", "3", ",", " ", "1", "7", ",", " ", "1", "9",
            ],
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
        );
    }

    #[test]
    fn case08_code_tip() {
        assert(
            &[
                "Code",
                " tip",
                ":",
                " avoid",
                " unwrap",
                "()",
                " in",
                " production",
                " Rust",
                ".",
            ],
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
        );
    }

    #[test]
    fn case09_compute() {
        assert(
            &[
                "Compute", ":", " ", "1", "2", "7", " *", " ", "4", "3", " =", " ", "5", "4", "6",
                "1",
            ],
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
        );
    }

    #[test]
    fn case10_python_code() {
        assert(
            &[
                "Code", ":\n", "```", "python", "\n", "for", " i", " in", " range", "(", "3",
                "):\n", "   ", " print", "(i", ")\n", "```",
            ],
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
        );
    }
}
