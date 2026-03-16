use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::error::AppError;

pub struct VocabEngine {
    vocab: HashMap<String, u32>,
}

impl VocabEngine {
    pub fn new(vocab_file: File) -> Result<Self, AppError> {
        let reader = BufReader::new(vocab_file);
        let mut vocab = HashMap::with_capacity(151644);

        for line in reader.lines() {
            let line = line.map_err(AppError::Io)?;
            let line = line.trim();
            if !line.starts_with('"') {
                continue;
            }
            if let Some((key, id)) = parse_json_string(line) {
                vocab.insert(key, id);
            }
        }

        Ok(Self { vocab })
    }

    pub fn tokenize(&self, word: &str) -> Result<u32, AppError> {
        self.vocab
            .get(word)
            .copied()
            .ok_or(AppError::InvalidState("token not found in vocab"))
    }
}

fn parse_json_string(s: &str) -> Option<(String, u32)> {
    debug_assert!(s.starts_with('"'));
    let inner = &s[1..];
    let mut result = String::new();
    let mut iter = inner.char_indices();
    loop {
        let (i, c) = iter.next()?;
        match c {
            '"' => {
                let rest = inner[i + 1..].trim_start();
                let rest = rest.strip_prefix(':')?;
                let id = rest
                    .trim()
                    .trim_end_matches(',')
                    .trim()
                    .parse::<u32>()
                    .ok()?;
                return Some((result, id));
            }
            '\\' => match iter.next()?.1 {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'u' => {
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        hex.push(iter.next()?.1);
                    }
                    result.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                other => {
                    result.push('\\');
                    result.push(other);
                }
            },
            _ => result.push(c),
        }
    }
}
