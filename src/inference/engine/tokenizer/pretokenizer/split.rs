use super::Error;

pub struct SplitEngine;

// behavior=Isolated, invert=false, regex_pattern:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
impl SplitEngine {
    pub fn new() -> Result<Self, Error> {
        Ok(Self)
    }

    pub fn pretokenize<'a>(&self, input: &'a str) -> Result<Vec<&'a str>, Error> {
        let mut slices = Vec::new();
        let mut remaining = input;

        while !remaining.is_empty() {
            let prev_len = remaining.len();
            let token = next_token_slice(remaining);
            if token.is_empty() {
                debug_assert!(false, "split pretokenizer produced empty token");
                break;
            }

            slices.push(token);
            remaining = &remaining[token.len()..];

            debug_assert!(
                remaining.len() < prev_len,
                "split pretokenizer made no progress"
            );
        }

        debug_assert!(remaining.is_empty(), "split pretokenizer got stuck");

        Ok(slices)
    }
}

fn next_token_slice(input: &str) -> &str {
    debug_assert!(
        !input.is_empty(),
        "next_token_slice called with empty input"
    );
    if input.is_empty() {
        return "";
    }

    if let Some(token) = match_contraction(input) {
        return token;
    }

    if let Some(token) = match_letter_chunk(input) {
        return token;
    }

    let first = char_at(input, 0).expect("next_token_slice is only called with non-empty string");

    if first.is_numeric() {
        return &input[..first.len_utf8()];
    }

    if let Some(token) = match_symbol_chunk(input) {
        return token;
    }

    if let Some(token) = match_newline_chunk(input) {
        return token;
    }

    if let Some(token) = match_whitespace_before_nonspace(input) {
        return token;
    }

    if first.is_whitespace() {
        return consume_whitespace(input);
    }

    panic!("next_token_slice fell through; forcing single-char token");
}

fn match_contraction(input: &str) -> Option<&str> {
    if char_at(input, 0) != Some('\'') {
        return None;
    }

    let patterns = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    for pattern in patterns {
        let candidate = input.get(..pattern.len())?;
        if candidate.eq_ignore_ascii_case(pattern) {
            return Some(candidate);
        }
    }

    None
}

fn match_letter_chunk(input: &str) -> Option<&str> {
    let first = char_at(input, 0)?;
    let mut prefix_len = 0;

    if !is_newline(first) && !first.is_alphabetic() && !first.is_numeric() {
        prefix_len = first.len_utf8();
    }

    let letters = consume_letters(&input[prefix_len..]);
    if !letters.is_empty() {
        Some(&input[..prefix_len + letters.len()])
    } else {
        None
    }
}

fn match_symbol_chunk(input: &str) -> Option<&str> {
    let first = char_at(input, 0)?;
    let mut prefix_len = 0;

    if first == ' ' {
        prefix_len = first.len_utf8();
        if !is_symbol_char(char_at(input, prefix_len)?) {
            return None;
        }
    } else if !is_symbol_char(first) {
        return None;
    }

    let symbols = consume_symbols(&input[prefix_len..]);
    if symbols.is_empty() {
        return None;
    }

    let rest = &input[prefix_len + symbols.len()..];
    let newlines = consume_newlines(rest);
    Some(&input[..prefix_len + symbols.len() + newlines.len()])
}

fn match_newline_chunk(input: &str) -> Option<&str> {
    let inline_ws = consume_inline_whitespace(input);
    let rest = &input[inline_ws.len()..];

    if !is_newline(char_at(rest, 0)?) {
        return None;
    }

    let newlines = consume_newlines(rest);
    Some(&input[..inline_ws.len() + newlines.len()])
}

fn match_whitespace_before_nonspace(input: &str) -> Option<&str> {
    if !char_at(input, 0)?.is_whitespace() {
        return None;
    }

    let mut idx = 0;
    let mut last_boundary = 0;

    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() {
            break;
        }
        last_boundary = idx;
        idx += ch.len_utf8();
    }

    if idx == 0 || idx == input.len() {
        return None;
    }

    if last_boundary > 0 {
        Some(&input[..last_boundary])
    } else {
        None
    }
}

fn consume_letters(input: &str) -> &str {
    let mut idx = 0;
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_alphabetic() {
            break;
        }
        idx += ch.len_utf8();
    }
    &input[..idx]
}

fn consume_symbols(input: &str) -> &str {
    let mut idx = 0;
    while let Some(ch) = char_at(input, idx) {
        if !is_symbol_char(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    &input[..idx]
}

fn consume_newlines(input: &str) -> &str {
    let mut idx = 0;
    while let Some(ch) = char_at(input, idx) {
        if !is_newline(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    &input[..idx]
}

fn consume_inline_whitespace(input: &str) -> &str {
    let mut idx = 0;
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() || is_newline(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    &input[..idx]
}

fn consume_whitespace(input: &str) -> &str {
    let mut idx = 0;
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    &input[..idx]
}

fn is_newline(ch: char) -> bool {
    ch == '\r' || ch == '\n'
}

fn is_symbol_char(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_numeric()
}

fn char_at(input: &str, idx: usize) -> Option<char> {
    input.get(idx..)?.chars().next()
}

#[cfg(test)]
mod tests {
    use super::SplitEngine;

    fn assert_slices(input: &str, expected: &[&str]) {
        let engine = SplitEngine::new().unwrap();
        let slices = engine
            .pretokenize(input)
            .expect("split pretokenize should succeed");
        assert_eq!(
            slices, expected,
            "Pretokenizing of {:?} failed: pretokenized:{:?}, expected:{:?}",
            input, slices, expected
        );
    }

    #[test]
    fn split_case1_hello() {
        assert_slices("Hello!", &["Hello", "!"]);
    }

    #[test]
    fn split_case2_summarize() {
        assert_slices(
            "Summarize: Rust ownership prevents data races.",
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
        );
    }

    #[test]
    fn split_case3_math_question() {
        assert_slices(
            "What is 2 + 2?",
            &["What", " is", " ", "2", " +", " ", "2", "?"],
        );
    }

    #[test]
    fn split_case4_whitespace() {
        assert_slices(
            "Whitespace test:  keep   multiple spaces, tabs\t, and blank lines\n\nend.",
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
        );
    }

    #[test]
    fn split_case5_emoji() {
        assert_slices(
            "Emoji test: cats 😺 rockets 🚀 and sparkles ✨.",
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
        );
    }

    #[test]
    fn split_case6_json() {
        assert_slices(
            "Answer with JSON: { \"name\": \"Alice\", \"age\": 27 }",
            &[
                "Answer", " with", " JSON", ":", " {", " \"", "name", "\":", " \"", "Alice", "\",",
                " \"", "age", "\":", " ", "2", "7", " }",
            ],
        );
    }

    #[test]
    fn split_case7_list_numbers() {
        assert_slices(
            "List these: 2, 3, 5, 7, 11, 13, 17, 19",
            &[
                "List", " these", ":", " ", "2", ",", " ", "3", ",", " ", "5", ",", " ", "7", ",",
                " ", "1", "1", ",", " ", "1", "3", ",", " ", "1", "7", ",", " ", "1", "9",
            ],
        );
    }

    #[test]
    fn split_case8_code_tip() {
        assert_slices(
            "Code tip: avoid unwrap() in production Rust.",
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
        );
    }

    #[test]
    fn split_case9_compute() {
        assert_slices(
            "Compute: 127 * 43 = 5461",
            &[
                "Compute", ":", " ", "1", "2", "7", " *", " ", "4", "3", " =", " ", "5", "4", "6",
                "1",
            ],
        );
    }

    #[test]
    fn split_case10_python_block() {
        assert_slices(
            "Code:\n```python\nfor i in range(3):\n    print(i)\n```",
            &[
                "Code", ":\n", "```", "python", "\n", "for", " i", " in", " range", "(", "3",
                "):\n", "   ", " print", "(i", ")\n", "```",
            ],
        );
    }
}
