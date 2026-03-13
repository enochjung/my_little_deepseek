use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct MergeEngine {
    // Maps (left, right) pair -> merge rank (index in merges list)
    pair_to_rank: HashMap<(String, String), u32>,
}

impl MergeEngine {
    pub fn new(merges_file: File) -> Self {
        let mut pair_to_rank = HashMap::new();
        let mut rank: u32 = 0;
        for line in BufReader::new(merges_file).lines() {
            let line = line.expect("failed to read merges line");
            let line = line.trim();
            // Array element lines look like: "left right",
            if !line.starts_with('"') {
                continue;
            }
            // Parse as a JSON string (with unescaping), then split on the first space
            if let Some((merged, _)) = parse_json_string(line) {
                if let Some(pos) = merged.find(' ') {
                    let left = merged[..pos].to_string();
                    let right = merged[pos + 1..].to_string();
                    pair_to_rank.insert((left, right), rank);
                    rank += 1;
                }
            }
        }
        Self { pair_to_rank }
    }

    fn rank(&self, left: &str, right: &str) -> Option<u32> {
        self.pair_to_rank
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    /// Apply BPE merges to a sequence of initial symbols (one char each).
    /// Returns the merged symbol sequence.
    pub fn merge(&self, symbols: &mut Vec<String>) {
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.rank(&w[0], &w[1]).map(|r| (r, i)))
                .min_by_key(|&(rank, _)| rank);
            let Some((_, pos)) = best else { break };
            let merged = symbols[pos].clone() + &symbols[pos + 1];
            symbols[pos] = merged;
            symbols.remove(pos + 1);
        }
    }
}

/// Parse a JSON-quoted string from the start of `s`.
/// Returns `(unescaped_content, remaining_after_closing_quote)`.
fn parse_json_string(s: &str) -> Option<(String, &str)> {
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
