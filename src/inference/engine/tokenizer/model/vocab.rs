use super::{Error, ModelData};
use crate::inference::data::*;
use std::collections::HashMap;

pub struct VocabEngine<'a> {
    #[allow(unused)]
    model_data: &'a ModelData,
    vocab_map: VocabMap,
}

impl<'a> VocabEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let mut vocab_map = VocabMap::new();

        match (&model_data.vocab_binary, &model_data.vocab_text) {
            (Some(vocab_binary), _) => parse_vocab_binary(vocab_binary, &mut vocab_map),
            (None, Some(vocab_text)) => parse_vocab_text(vocab_text, &mut vocab_map),
            (None, None) => return Err(Error::data_not_provided("vocab")),
        }?;

        Ok(Self {
            model_data,
            vocab_map,
        })
    }

    pub fn tokenize(&self, word: &str) -> Result<u32, Error> {
        if self.vocab_map.get(word).is_none() {
            panic!("none. word:`{}` {:?}", word, word.as_bytes())
        }
        Ok(self.vocab_map.get(word).unwrap())
    }
}

#[allow(unused)]
fn parse_vocab_binary(vocab_binary: &VocabBinary, vocab_map: &mut VocabMap) -> Result<(), Error> {
    todo!("parsing vocab binary is not implemented yet");
}

fn parse_vocab_text(vocab_text: &VocabText, vocab_map: &mut VocabMap) -> Result<(), Error> {
    let iter = vocab_text.parse()?;

    for line in iter {
        let line = line?;
        let (token, id) = line;
        vocab_map.insert(token, id);
    }

    Ok(())
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
