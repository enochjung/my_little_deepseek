use crate::error::AppError;

pub struct SplitEngine;

// behavior=Isolated, invert=false, regex_pattern:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
impl SplitEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn pretokenize(&self, input: &str) -> Result<Vec<(usize, usize)>, AppError> {
        let mut spans = Vec::with_capacity(input.len().min(32));
        let mut start = 0;

        while let Some((token_start, token_end)) = next_token_span(input, start) {
            if token_start != start || token_end <= token_start {
                return Err(AppError::InvalidState(
                    "split pretokenizer made no progress",
                ));
            }

            spans.push((token_start, token_end));
            start = token_end;
        }

        if start == input.len() {
            Ok(spans)
        } else {
            Err(AppError::InvalidState("split pretokenizer got stuck"))
        }
    }
}

fn next_token_span(input: &str, start: usize) -> Option<(usize, usize)> {
    if start >= input.len() {
        return None;
    }

    if let Some(end) = match_contraction(input, start) {
        return Some((start, end));
    }

    if let Some(end) = match_letter_chunk(input, start) {
        return Some((start, end));
    }

    let first = char_at(input, start)?;

    if first.is_numeric() {
        return Some((start, start + first.len_utf8()));
    }

    if let Some(end) = match_symbol_chunk(input, start) {
        return Some((start, end));
    }

    if let Some(end) = match_newline_chunk(input, start) {
        return Some((start, end));
    }

    if let Some(end) = match_whitespace_before_nonspace(input, start) {
        return Some((start, end));
    }

    if first.is_whitespace() {
        return Some((start, consume_whitespace(input, start)));
    }

    None
}

fn match_contraction(input: &str, start: usize) -> Option<usize> {
    if char_at(input, start) != Some('\'') {
        return None;
    }

    let rest = input.get(start..)?;
    let patterns = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];

    for pattern in patterns {
        let candidate = rest.get(..pattern.len())?;
        if candidate.eq_ignore_ascii_case(pattern) {
            return Some(start + pattern.len());
        }
    }

    None
}

fn match_letter_chunk(input: &str, start: usize) -> Option<usize> {
    let first = char_at(input, start)?;
    let mut idx = start;

    if !is_newline(first) && !first.is_alphabetic() && !first.is_numeric() {
        idx += first.len_utf8();
    }

    let letters_end = consume_letters(input, idx);
    if letters_end > idx {
        Some(letters_end)
    } else {
        None
    }
}

fn match_symbol_chunk(input: &str, start: usize) -> Option<usize> {
    let first = char_at(input, start)?;
    let mut idx = start;

    if first == ' ' {
        idx += first.len_utf8();
        if !is_symbol_char(char_at(input, idx)?) {
            return None;
        }
    } else if !is_symbol_char(first) {
        return None;
    }

    let symbol_end = consume_symbols(input, idx);
    if symbol_end == idx {
        return None;
    }

    Some(consume_newlines(input, symbol_end))
}

fn match_newline_chunk(input: &str, start: usize) -> Option<usize> {
    let mut idx = consume_inline_whitespace(input, start);
    if !is_newline(char_at(input, idx)?) {
        return None;
    }

    idx = consume_newlines(input, idx);
    Some(idx)
}

fn match_whitespace_before_nonspace(input: &str, start: usize) -> Option<usize> {
    if !char_at(input, start)?.is_whitespace() {
        return None;
    }

    let mut idx = start;
    let mut last_boundary = start;

    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() {
            break;
        }
        last_boundary = idx;
        idx += ch.len_utf8();
    }

    if idx == start || idx == input.len() {
        return None;
    }

    if last_boundary > start {
        Some(last_boundary)
    } else {
        None
    }
}

fn consume_letters(input: &str, mut idx: usize) -> usize {
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_alphabetic() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_symbols(input: &str, mut idx: usize) -> usize {
    while let Some(ch) = char_at(input, idx) {
        if !is_symbol_char(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_newlines(input: &str, mut idx: usize) -> usize {
    while let Some(ch) = char_at(input, idx) {
        if !is_newline(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_inline_whitespace(input: &str, mut idx: usize) -> usize {
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() || is_newline(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_whitespace(input: &str, mut idx: usize) -> usize {
    while let Some(ch) = char_at(input, idx) {
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
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

    fn assert_spans(input: &str, expected: &[(usize, usize)]) {
        let engine = SplitEngine::new();
        let spans = engine
            .pretokenize(input)
            .expect("split pretokenize should succeed");
        assert_eq!(spans, expected);
    }

    #[test]
    fn split_case1_hello() {
        assert_spans("Hello!", &[(0, 5), (5, 6)]);
    }

    #[test]
    fn split_case2_summarize() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case3_math_question() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case4_whitespace() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case5_emoji() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case6_json() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case7_list_numbers() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case8_code_tip() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case9_compute() {
        assert_spans(
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
        );
    }

    #[test]
    fn split_case10_python_block() {
        assert_spans(
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
        );
    }
}
