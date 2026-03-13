use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;

pub struct VocabEngine {
    token_to_id: HashMap<String, u32>,
}

impl VocabEngine {
    pub fn new(vocab_file: File) -> Self {
        let mut token_to_id = HashMap::new();
        for line in BufReader::new(vocab_file).lines() {
            let line = line.expect("failed to read vocab line");
            let line = line.trim();
            // lines look like: "<token>": <id>,
            if !line.starts_with('"') {
                continue;
            }
            if let Some((key, rest)) = parse_json_string_key(line) {
                // rest is like ": 12345," or ": 12345"
                let rest = rest.trim_start_matches(':').trim();
                let id_str = rest.trim_end_matches(',').trim();
                if let Ok(id) = id_str.parse::<u32>() {
                    token_to_id.insert(key, id);
                }
            }
        }
        Self { token_to_id }
    }

    /// Map a sequence of BPE symbols to their vocab IDs.
    /// Returns `None` if any symbol is not in the vocabulary.
    pub fn encode(&self, symbols: &[String]) -> Option<Vec<u32>> {
        symbols.iter().map(|s| self.token_to_id.get(s.as_str()).copied()).collect()
    }
}

/// Parse a JSON-quoted string key from the start of `s`.
/// Returns `(unescaped_key, remaining_after_closing_quote)`.
fn parse_json_string_key(s: &str) -> Option<(String, &str)> {
    let s = s.strip_prefix('"')?;
    let mut result = String::new();
    let mut chars = s.char_indices();
    loop {
        let (i, c) = chars.next()?;
        if c == '"' {
            return Some((result, &s[i + 1..]));
        } else if c == '\\' {
            let (_, esc) = chars.next()?;
            match esc {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                other => {
                    result.push('\\');
                    result.push(other);
                }
            }
        } else {
            result.push(c);
        }
    }
}
