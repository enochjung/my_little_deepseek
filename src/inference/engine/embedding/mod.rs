use crate::inference::data::*;
use crate::inference::{Error, ModelData, Tensor};

const BF16_BYTES: usize = 2;

#[allow(unused)]
pub struct EmbeddingEngine<'a> {
    _model_data: &'a ModelData,
    embed_tokens_info: &'a TensorInfo,
    embed_data: &'a [u8],
}

impl<'a> EmbeddingEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let (embed_tokens_info, embed_data) =
            match (&model_data.weight_binary, &model_data.weight_text) {
                (Some(weight_binary), _) => parse_embed_tokens_binary(weight_binary),
                (None, Some(weight_text)) => parse_embed_tokens_text(weight_text),
                (None, None) => return Err(Error::data_not_provided("weight")),
            }?;

        Ok(Self {
            _model_data: model_data,
            embed_tokens_info,
            embed_data,
        })
    }

    pub fn embed(&self, token_ids: &[u32]) -> Result<Tensor<'a>, Error> {
        if token_ids.is_empty() {
            return Err(Error::empty_token_ids());
        }

        // let vocab_size = self.embed_tokens_info.shape[0] as usize;
        let hidden_size = self.embed_tokens_info.shape[1] as usize;
        let row_bytes = hidden_size * BF16_BYTES;

        let mut output = Vec::with_capacity(token_ids.len() * row_bytes);

        for &token_id in token_ids {
            let token_idx = token_id as usize;

            // TODO: check row / col major
            let start = token_idx * row_bytes;
            let end = start + row_bytes;
            output.extend_from_slice(&self.embed_data[start..end]);
        }

        Tensor::owned(output, [token_ids.len(), hidden_size])
    }
}

fn parse_embed_tokens_binary<'a>(
    _weight_binary: &'a WeightBinary,
) -> Result<(&'a TensorInfo, &'a [u8]), Error> {
    todo!("parsing unicode binary is not implemented yet");
}

fn parse_embed_tokens_text<'a>(
    weight_text: &'a WeightText,
) -> Result<(&'a TensorInfo, &'a [u8]), Error> {
    let (weight_info, payload) = weight_text.parse()?;

    let embed_tokens_info = &weight_info.embed_tokens_weight;

    let start = embed_tokens_info.offset.start as usize;
    let end = embed_tokens_info.offset.end as usize;
    let embed_data = &payload[start..end];

    Ok((embed_tokens_info, embed_data))
}
