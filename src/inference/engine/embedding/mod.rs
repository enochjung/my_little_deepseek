mod rotary_position_embedding;
mod word_embedding;

use crate::inference::{Error, ModelData, tensor};
use word_embedding::WordEmbeddingEngine;

#[allow(unused)]
pub struct EmbeddingEngine<'a, D: tensor::DataType> {
    _model_data: &'a ModelData,
    word_embedding_engine: WordEmbeddingEngine<'a, D>,
}

impl<'a> EmbeddingEngine<'a, tensor::F32> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let word_embedding_engine = WordEmbeddingEngine::new(model_data)?;

        Ok(Self {
            _model_data: model_data,
            word_embedding_engine,
        })
    }

    pub fn word_embed(
        &'a self,
        token_id: u32,
    ) -> Result<tensor::Tensor<tensor::F32, tensor::HostMemoryRef<'a>>, Error> {
        self.word_embedding_engine.word_embed(token_id)
    }
}
