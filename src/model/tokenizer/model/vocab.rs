use crate::config::{Format, VocabFormat};
use std::collections::HashMap;

pub(crate) struct VocabEngine {
    vocab_map: VocabMap,
}

impl VocabEngine {
    pub(crate) fn new(vocab_format: &VocabFormat) -> Result<Self, crate::Error> {
        let mut vocab_map = VocabMap::new();

        let (file, parse) = vocab_format.read()?;
        let iter = parse(&file);

        for line in iter {
            let line = line?;
            let (token, id) = line;
            vocab_map.insert(token, id);
        }

        Ok(Self { vocab_map })
    }

    pub fn tokenize(&self, word: &str) -> Result<u32, crate::Error> {
        if self.vocab_map.get(word).is_none() {
            panic!("none. word:`{}` {:?}", word, word.as_bytes())
        }
        Ok(self.vocab_map.get(word).unwrap())
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
