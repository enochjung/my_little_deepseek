use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::error::AppError;

pub struct MergesEngine {
    // Key: "left\x1eright", value: merge rank (lower = higher priority)
    merges: HashMap<String, usize>,
}

impl MergesEngine {
    pub fn new(merges_file: File) -> Result<Self, AppError> {
        let reader = BufReader::new(merges_file);
        let mut merges = HashMap::with_capacity(151388);
        let mut rank = 0usize;

        for line in reader.lines() {
            let line = line.map_err(AppError::Io)?;
            let line = line.trim();
            if !line.starts_with('"') {
                continue;
            }
            if let Some((left, right)) = parse_json_string(line) {
                let key = make_key(&left, &right);
                merges.insert(key, rank);
                rank += 1;
            }
        }

        Ok(Self { merges })
    }

    pub fn merge(&self, list: &[String]) -> Result<Vec<String>, AppError> {
        let mut tokens: Vec<String> = list.to_vec();

        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..tokens.len() - 1 {
                let key = make_key(&tokens[i], &tokens[i + 1]);
                if let Some(&rank) = self.merges.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let left = tokens[best_idx].clone();
            let right = tokens[best_idx + 1].clone();
            let merged = format!("{}{}", left, right);

            let mut new_tokens = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                    new_tokens.push(merged.clone());
                    i += 2;
                } else {
                    new_tokens.push(std::mem::take(&mut tokens[i]));
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        Ok(tokens)
    }
}

#[inline]
fn make_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + 1 + right.len());
    key.push_str(left);
    key.push('\x1e');
    key.push_str(right);
    key
}

fn parse_json_string(s: &str) -> Option<(String, String)> {
    debug_assert!(s.starts_with('"'));
    let inner = &s[1..];
    let mut result = String::new();
    let mut iter = inner.char_indices();
    loop {
        let (_, c) = iter.next()?;
        match c {
            '"' => {
                let pos = result.find(' ')?;
                let right = result[pos + 1..].to_string();
                result.truncate(pos);
                return Some((result, right));
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
