use core::{ElemType, MLTError, Memory, MemoryMut};

use crate::rms_norm::RMSNorm;
use crate::tensor::Tensor;

pub struct FeedForward<T: ElemType, M: Memory<T>> {
    rms_norm: RMSNorm<T, M>,
    gate: Tensor<T, M>,
    up: Tensor<T, M>,
    down: Tensor<T, M>,
}

impl<T: ElemType, M: Memory<T>> FeedForward<T, M> {
    pub fn new(
        norm: Tensor<T, M>,
        gate: Tensor<T, M>,
        up: Tensor<T, M>,
        down: Tensor<T, M>,
        rms_norm_epsilon: T,
    ) -> Self {
        let rms_norm = RMSNorm::new(norm, rms_norm_epsilon);
        Self {
            rms_norm,
            gate: gate.transpose(),
            up: up.transpose(),
            down: down.transpose(),
        }
    }

    pub fn execute<D: MemoryMut<T, Base = M::Base>, T0: MemoryMut<T, Base = M::Base>>(
        &self,
        t: u32,
        target_t_x_h: &mut Tensor<T, D>,
        tmp_2t_x_i: &mut Tensor<T, T0>,
    ) -> Result<(), MLTError> {
        let (mut gate_t_x_i, mut up_t_x_i) = tmp_2t_x_i.split_row(t)?;
        self.rms_norm.execute(target_t_x_h)?;
        gate_t_x_i.matmul(target_t_x_h, &self.gate)?;
        up_t_x_i.matmul(target_t_x_h, &self.up)?;
        gate_t_x_i.silu();
        gate_t_x_i.elem_mul_assign(&up_t_x_i)?;
        target_t_x_h.matmul(&gate_t_x_i, &self.down)?;
        Ok(())
    }
}
