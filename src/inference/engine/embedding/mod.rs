mod word_embedding;

use crate::inference::{Error, ModelData, tensor};
use word_embedding::WordEmbeddingEngine;

pub struct EmbeddingEngine<'a> {
    _model_data: &'a ModelData,
    word_embedding_engine: WordEmbeddingEngine<'a>,
}

impl<'a> EmbeddingEngine<'a> {
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
    ) -> Result<tensor::TensorRef<'a, tensor::F32>, Error> {
        self.word_embedding_engine.word_embed(token_id)
    }
}
