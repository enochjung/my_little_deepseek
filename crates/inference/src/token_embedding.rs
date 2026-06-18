use core::{ElemType, MLTError, Memory};

use crate::tensor::Tensor;

pub struct TokenEmbedding<T: ElemType, M: Memory<T>> {
    embed: Tensor<T, M>,
    vocab_size: u32,
    hidden_size: u32,
}

impl<T: ElemType, M: Memory<T>> TokenEmbedding<T, M> {
    pub fn new(embed: Tensor<T, M>, vocab_size: u32, hidden_size: u32) -> Self {
        Self {
            embed,
            vocab_size,
            hidden_size,
        }
    }

    pub fn execute(&self, token_id: u32) -> Result<Tensor<T, &M::Base>, MLTError> {
        if token_id >= self.vocab_size {
            return Err(MLTError::out_of_bound(
                token_id as usize,
                self.vocab_size as usize,
            ));
        }
        Ok(self
            .embed
            .slice(token_id..token_id + 1, 0..self.hidden_size))
    }
}
