mod kv_cache;
mod special_token;

use crate::Model;
pub(crate) use kv_cache::KVCache;
use std::marker::PhantomData;

mod private {
    pub trait SessionState {}
}

pub struct Ready;
impl private::SessionState for Ready {}

pub struct Decoding;
impl private::SessionState for Decoding {}

pub struct Session<'a, T: private::SessionState> {
    model: &'a Model,
    tokens: Vec<u32>,
    kv_cache: KVCache,
    _phantom: PhantomData<T>,
}

impl<'a> Session<'a, Ready> {
    pub fn send_prompt(self, user_input: &str) -> Session<'a, Decoding> {
        todo!()
        /*
        self.tokens.push(special_token::USER);

        let mut input_tokens = self.model.tokenize(user_input)?;
        self.tokens.append(&mut input_tokens);

        self.tokens.push(special_token::ASSISTANT);
        self.tokens.push(special_token::THINK_START);

        let embedded_tensor = self.model.build_embedding_vectors(&self.tokens)?;
        */
    }

    pub(crate) fn new(model: &'a Model) -> Self {
        let mut tokens = Vec::new();
        tokens.push(special_token::BEGIN_OF_SENTENCE);
        let kv_cache = KVCache::new();

        Self {
            model,
            tokens,
            kv_cache,
            _phantom: PhantomData,
        }
    }
}

impl<'a> Session<'a, Decoding> {
    pub fn start(&mut self) -> () {
        todo!()
    }

    pub fn pause(&mut self) -> () {
        todo!()
    }

    pub fn is_running(&self) -> bool {
        todo!()
    }

    pub fn is_done(&self) -> bool {
        todo!()
    }

    pub fn finish_decoding(self) -> Session<'a, Ready> {
        todo!()
    }

    pub fn get_next_string(&mut self) -> Option<String> {
        todo!()
    }
}
