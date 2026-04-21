use crate::inference::data::*;
use crate::inference::tensor;
use crate::inference::{Error, ModelData};

use tensor::{BF16, DataType, F32, HostMemory, HostMemoryRef, Tensor};

pub struct WordEmbeddingEngine<'a, D: DataType> {
    _model_data: &'a ModelData,
    table: Tensor<D, HostMemory>,
}

impl<'a> WordEmbeddingEngine<'a, F32> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let tensor = match (&model_data.weight_binary, &model_data.weight_text) {
            (Some(weight_binary), _) => parse_embed_tokens_binary(weight_binary),
            (None, Some(weight_text)) => parse_embed_tokens_text(weight_text),
            (None, None) => return Err(Error::data_not_provided("weight")),
        }?;

        let table = Tensor::<F32, HostMemory>::from(&tensor);

        Ok(Self {
            _model_data: model_data,
            table,
        })
    }

    pub fn word_embed<'b>(
        &'b self,
        token_id: u32,
    ) -> Result<Tensor<F32, HostMemoryRef<'b>>, Error> {
        let [nrow, ncol] = self.table.shape();
        let row = token_id as usize;
        if row >= nrow {
            return Err(Error::out_of_bound(row, nrow));
        }

        self.table.slice(row..(row + 1), 0..ncol)
    }
}

fn parse_embed_tokens_binary<'a>(
    _weight_binary: &'a WeightBinary,
) -> Result<Tensor<BF16, HostMemoryRef<'a>>, Error> {
    todo!("parsing unicode binary is not implemented yet");
}

fn parse_embed_tokens_text<'a>(
    weight_text: &'a WeightText,
) -> Result<Tensor<BF16, HostMemoryRef<'a>>, Error> {
    let (weight_info, payload) = weight_text.parse()?;
    let embed_tokens_info = &weight_info.embed_tokens_weight;

    let start = embed_tokens_info.offset.start as usize;
    let end = embed_tokens_info.offset.end as usize;
    let data = &payload[start..end];
    let nrow = embed_tokens_info.shape[0] as usize; // 151936
    let ncol = embed_tokens_info.shape[1] as usize; // 1536
    let tensor = Tensor::<BF16, HostMemoryRef>::new(data, nrow, ncol, false)?;

    Ok(tensor)
}
