use crate::config::{Format, VocabFormat};
use std::collections::HashMap;

pub(crate) struct Vocab {
    vocab_map: VocabMap,
    inverse_vocab_map: InverseVocabMap,
}

impl Vocab {
    pub(crate) fn new(vocab_format: &VocabFormat) -> Result<Self, crate::Error> {
        let mut vocab_map = VocabMap::new();
        let mut inverse_vocab_map = InverseVocabMap::new();

        let (file, parse) = vocab_format.read()?;
        let iter = parse(&file);

        for line in iter {
            let line = line?;
            let (token, id) = line;
            inverse_vocab_map.insert(id, &token);
            vocab_map.insert(token, id);
        }

        Ok(Self {
            vocab_map,
            inverse_vocab_map,
        })
    }

    pub(crate) fn encode(&self, word: &str) -> Result<u32, crate::Error> {
        if self.vocab_map.get(word).is_none() {
            panic!("none. word:`{}` {:?}", word, word.as_bytes())
        }
        Ok(self.vocab_map.get(word).unwrap())
    }

    pub(crate) fn decode(&self, tokens: &[u32]) -> Result<(usize, String), crate::Error> {
        let mut byte_buffer = Vec::new();

        for (idx, &token_id) in tokens.iter().enumerate() {
            if let Some(bytes) = self.inverse_vocab_map.get(&token_id) {
                byte_buffer.extend(bytes);
            }

            match std::str::from_utf8(&byte_buffer) {
                Ok(valid_str) => {
                    return Ok((idx + 1, valid_str.to_string()));
                }
                Err(_) => {
                    continue;
                }
            }
        }

        Ok((
            tokens.len(),
            String::from_utf8_lossy(&byte_buffer).into_owned(),
        ))
    }
}

struct VocabMap {
    map: HashMap<String, u32>,
}

impl VocabMap {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(151643),
        }
    }

    fn get(&self, key: &str) -> Option<u32> {
        self.map.get(key).copied()
    }

    fn insert(&mut self, key: String, id: u32) -> () {
        self.map.insert(key, id);
    }
}

struct InverseVocabMap {
    map: HashMap<u32, Vec<u8>>,
}

impl InverseVocabMap {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(151643),
        }
    }

    fn get(&self, key: &u32) -> Option<&Vec<u8>> {
        self.map.get(key)
    }

    fn insert(&mut self, key: u32, text: &str) -> () {
        let is_special_token = text.chars().any(|c| (c as u32) > 323);

        let bytes = if is_special_token {
            text.as_bytes().to_vec()
        } else {
            text.chars()
                .map(|c| {
                    let b = c as u32;
                    if (256..=288).contains(&b) {
                        (b - 256) as u8
                    } else if (289..=322).contains(&b) {
                        (b - 289 + 127) as u8
                    } else if b == 323 {
                        173 as u8
                    } else {
                        b as u8
                    }
                })
                .collect()
        };

        self.map.insert(key, bytes);
    }
}
