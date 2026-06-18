use core::{BackendOps, ElemType, MLTError, Memory, MemoryMut, MemoryOwn};

use crate::rms_norm::RMSNorm;
use crate::tensor::Tensor;

pub struct Sampling<T: ElemType, M: Memory<T>> {
    rms_norm: RMSNorm<T, M>,
    lm_head: Tensor<T, M>,
}

impl<T: ElemType, M: Memory<T>> Sampling<T, M>
where
    <M::Base as MemoryOwn<T>>::Operator: BackendOps<T>,
{
    pub fn new(last_norm: Tensor<T, M>, lm_head: Tensor<T, M>, rms_norm_epsilon: f32) -> Self {
        let rms_norm = RMSNorm::new(last_norm, rms_norm_epsilon);
        Self {
            rms_norm,
            lm_head: lm_head.transpose(),
        }
    }

    pub fn execute<D: MemoryMut<T, Base = M::Base>, T0: MemoryMut<T, Base = M::Base>>(
        &self,
        target_1_x_h: &mut Tensor<T, D>,
        tmp_1_x_v: &mut Tensor<T, T0>,
    ) -> Result<u32, MLTError> {
        self.rms_norm.execute(target_1_x_h)?;
        tmp_1_x_v.matmul(target_1_x_h, &self.lm_head)?;
        tmp_1_x_v.argmax()
    }
}
