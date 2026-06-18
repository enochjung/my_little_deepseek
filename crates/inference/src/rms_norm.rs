use core::{BackendOps, ElemType, MLTError, Memory, MemoryMut, MemoryOwn};

use crate::tensor::Tensor;

pub struct RMSNorm<T: ElemType, M: Memory<T>> {
    norm: Tensor<T, M>,
    epsilon: f32,
}

impl<T: ElemType, M: Memory<T>> RMSNorm<T, M>
where
    <M::Base as MemoryOwn<T>>::Operator: BackendOps<T>,
{
    pub fn new(norm: Tensor<T, M>, epsilon: f32) -> Self {
        Self { norm, epsilon }
    }

    pub fn execute<D: MemoryMut<T, Base = M::Base>>(
        &self,
        target_r_x_h: &mut Tensor<T, D>,
    ) -> Result<(), MLTError> {
        target_r_x_h.rms_norm(&self.norm, self.epsilon)
    }
}
