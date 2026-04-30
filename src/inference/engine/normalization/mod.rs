use super::{Error, ModelData, tensor};
use crate::inference::data::*;

use tensor::{BF16, F32, Tensor, TensorRef};

pub struct NormalizationEngine<'a> {
    _model_data: &'a ModelData,
    input_layernorm_weights: Vec<Tensor<F32>>,
    post_attention_layernorm_weights: Vec<Tensor<F32>>,
    norm_weight: Tensor<F32>,
}

impl<'a> NormalizationEngine<'a> {
    pub fn new(model_data: &'a ModelData) -> Result<Self, Error> {
        let (input_layernorm_weights, post_attention_layernorm_weights, norm_weight) =
            match (&model_data.weight_binary, &model_data.weight_text) {
                (Some(weight_binary), _) => parse_layernorms_binary(weight_binary),
                (None, Some(weight_text)) => parse_layernorms_text(weight_text),
                (None, None) => return Err(Error::data_not_provided("weight")),
            }?;

        let input_layernorm_weights = input_layernorm_weights
            .iter()
            .map(Tensor::<F32>::from)
            .collect();
        let post_attention_layernorm_weights = post_attention_layernorm_weights
            .iter()
            .map(Tensor::<F32>::from)
            .collect();
        let norm_weight = Tensor::<F32>::from(&norm_weight);

        Ok(Self {
            _model_data: model_data,
            input_layernorm_weights,
            post_attention_layernorm_weights,
            norm_weight,
        })
    }

    pub fn apply_input_rms_norm(
        &self,
        layer_idx: usize,
        input: &mut Tensor<F32>,
    ) -> Result<(), Error> {
        let weight_tensor = self.get_input_norm_tensor(layer_idx)?;
        let weight_ref = weight_tensor.as_ref(0..1, 0..weight_tensor.layout().ncol)?;
        input.rms_norm(&weight_ref)
    }

    #[allow(unused)]
    pub fn apply_post_attention_rms_norm(
        &self,
        layer_idx: usize,
        _input: &mut Tensor<F32>,
    ) -> Result<(), Error> {
        let _weight = self.get_post_norm_tensor(layer_idx)?;
        todo!("post-attention RMSNorm is not implemented yet")
    }

    #[allow(unused)]
    pub fn apply_final_rms_norm(&self, _input: &mut Tensor<F32>) -> Result<(), Error> {
        let _weight = self.get_final_norm_tensor();
        todo!("final RMSNorm is not implemented yet")
    }

    fn get_input_norm_tensor(&self, layer_idx: usize) -> Result<&Tensor<F32>, Error> {
        self.input_layernorm_weights
            .get(layer_idx)
            .ok_or_else(|| Error::out_of_bound(layer_idx, self.input_layernorm_weights.len()))
    }

    fn get_post_norm_tensor(&self, layer_idx: usize) -> Result<&Tensor<F32>, Error> {
        self.post_attention_layernorm_weights
            .get(layer_idx)
            .ok_or_else(|| {
                Error::out_of_bound(layer_idx, self.post_attention_layernorm_weights.len())
            })
    }

    fn get_final_norm_tensor(&self) -> &Tensor<F32> {
        &self.norm_weight
    }
}

fn parse_layernorms_binary<'a>(
    _weight_binary: &'a WeightBinary,
) -> Result<
    (
        Vec<TensorRef<'a, BF16>>,
        Vec<TensorRef<'a, BF16>>,
        TensorRef<'a, BF16>,
    ),
    Error,
> {
    todo!("parsing weight binary is not implemented yet");
}

fn parse_layernorms_text<'a>(
    weight_text: &'a WeightText,
) -> Result<
    (
        Vec<TensorRef<'a, BF16>>,
        Vec<TensorRef<'a, BF16>>,
        TensorRef<'a, BF16>,
    ),
    Error,
> {
    let (weight_info, payload) = weight_text.parse()?;

    let mut input_layernorm_weights = Vec::with_capacity(weight_info.layers.len());
    let mut post_attention_layernorm_weights = Vec::with_capacity(weight_info.layers.len());

    for layer_info in weight_info.layers.iter() {
        input_layernorm_weights.push(parse_layernorm_tensor(
            payload,
            &layer_info.input_layernorm_weight,
        )?);
        post_attention_layernorm_weights.push(parse_layernorm_tensor(
            payload,
            &layer_info.post_attention_layernorm_weight,
        )?);
    }

    let norm_weight = parse_layernorm_tensor(payload, &weight_info.norm_weight)?;

    Ok((
        input_layernorm_weights,
        post_attention_layernorm_weights,
        norm_weight,
    ))
}

fn parse_layernorm_tensor<'a>(
    _payload: &'a [u8],
    _tensor_info: &TensorInfo,
) -> Result<TensorRef<'a, BF16>, Error> {
    todo!()

    /*
    if tensor_info.shape.len() != 1 {
        return Err(Error::broken_data("weight", 0));
    }

    let start = tensor_info.offset.start as usize;
    let end = tensor_info.offset.end as usize;
    let data = &payload[start..end];
    let nrow = 1;
    let ncol = tensor_info.shape[0] as usize;

    Tensor::<BF16, HostMemoryRef>::new(data, nrow, ncol, false)
    */
}
