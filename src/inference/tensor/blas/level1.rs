use super::{DataType, F32, HostMemory};

pub struct OpVV<D: DataType, S>(core::marker::PhantomData<(D, S)>);

impl OpVV<F32, HostMemory> {
    pub fn rms(data: &[f32]) -> f32 {
        debug_assert!(!data.is_empty());

        let sq_sum = data.iter().map(|v| v * v).sum::<f32>();
        (sq_sum / (data.len() as f32)).sqrt()
    }

    pub fn mul(y: &mut [f32], x: &[f32], alpha: f32) {
        debug_assert_eq!(y.len(), x.len());

        for (y_value, x_value) in y.iter_mut().zip(x.iter()) {
            *y_value *= *x_value * alpha;
        }
    }
}
