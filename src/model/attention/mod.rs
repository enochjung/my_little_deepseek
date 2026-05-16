use super::AttentionLayerWeightInfo;
use super::RMSNormalizer;
use crate::storage::*;
use crate::tensor::*;

pub(crate) struct Attention<N: Numeric, S: Storage> {
    q_bias: TensorOwn<N, S>,
    q_weight: TensorOwn<N, S>,
    k_bias: TensorOwn<N, S>,
    k_weight: TensorOwn<N, S>,
    v_bias: TensorOwn<N, S>,
    v_weight: TensorOwn<N, S>,
    o_weight: TensorOwn<N, S>,
    rms_normalizer: RMSNormalizer<N, S>,
}

impl Attention<F32, Host> {
    pub(crate) fn new(
        weight_storage: &Host,
        attention_weight_info: &AttentionLayerWeightInfo,
    ) -> Result<Self, crate::Error> {
        let q_bias = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.q_proj_bias,
        ))?);
        let q_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.q_proj_weight,
        ))?);

        let k_bias = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.k_proj_bias,
        ))?);
        let k_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.k_proj_weight,
        ))?);

        let v_bias = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.v_proj_bias,
        ))?);
        let v_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.v_proj_weight,
        ))?);

        let o_weight = TensorOwn::<F32, Host>::from(&TensorRef::<BF16, Host>::try_from((
            weight_storage,
            &attention_weight_info.o_proj_weight,
        ))?);

        let rms_normalizer = RMSNormalizer::new(
            weight_storage,
            &attention_weight_info.input_layernorm_weight,
        )?;

        Ok(Self {
            q_bias,
            q_weight,
            k_bias,
            k_weight,
            v_bias,
            v_weight,
            o_weight,
            rms_normalizer,
        })
    }

    pub(crate) fn apply_attention(
        &self,
        target: &mut TensorOwn<F32, Host>,
        rms_norm_epsilon: f32,
    ) -> Result<(), crate::Error> {
        todo!()
        /*
        self.rms_normalizer.apply_rms_norm(target)?;

        let q = super::project_rows(target, &self.q_weight, Some(&self.q_bias), "attention_q")?;
        let k = super::project_rows(target, &self.k_weight, Some(&self.k_bias), "attention_k")?;
        let v = super::project_rows(target, &self.v_weight, Some(&self.v_bias), "attention_v")?;

        let mut output = super::project_rows(&q, &self.o_weight, None, "attention_output")?;

        let [rows, cols] = output.shape();
        let output_ptr = output.as_mut_ptr()? as *mut f32;
        let k_ptr = k.as_ptr() as *const f32;
        let v_ptr = v.as_ptr() as *const f32;
        let k_cols = k.shape()[1] as usize;
        let v_cols = v.shape()[1] as usize;

        for row_idx in 0..rows as usize {
            let k_mean = row_mean(unsafe { k_ptr.add(row_idx * k_cols) }, k_cols);
            let v_mean = row_mean(unsafe { v_ptr.add(row_idx * v_cols) }, v_cols);
            let blend = 0.5 * (k_mean + v_mean);
            let row_ptr = unsafe { output_ptr.add(row_idx * cols as usize) };

            for col_idx in 0..cols as usize {
                unsafe { *row_ptr.add(col_idx) += blend };
            }
        }

        Ok(output)
        */
    }
}
