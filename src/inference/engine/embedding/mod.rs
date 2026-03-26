use crate::inference::{Error, ModelData, Tensor};

#[allow(unused)]
pub struct EmbeddingEngine<'a> {
    weight: Tensor<'a>,
}

impl<'a> EmbeddingEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let _ = model_data;
        todo!()
    }

    #[allow(unused)]
    pub fn embed(&self, token_ids: &[u32]) -> Result<Tensor<'static>, Error> {
        let _ = token_ids;
        todo!()
    }
}
