use core::{ElemType, MLTError, Memory, MemoryMut, MemoryOwn};

use crate::kv_cache::KVCache;
use crate::rms_norm::RMSNorm;
use crate::rope::RoPE;
use crate::tensor::Tensor;

pub struct AttentionWeights<T: ElemType, M: Memory<T>> {
    pub q_bias: Tensor<T, M>,
    pub q_weight: Tensor<T, M>,
    pub k_bias: Tensor<T, M>,
    pub k_weight: Tensor<T, M>,
    pub v_bias: Tensor<T, M>,
    pub v_weight: Tensor<T, M>,
    pub o_weight: Tensor<T, M>,
}

pub struct GroupedQueryAttention<T: ElemType, M: Memory<T>> {
    rms_norm: RMSNorm<T, M>,
    qb: Tensor<T, M>,
    qw: Tensor<T, M>,
    kb: Tensor<T, M>,
    kw: Tensor<T, M>,
    vb: Tensor<T, M>,
    vw: Tensor<T, M>,
    ow: Tensor<T, M>,
    head_size: u32,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rope_theta: T,
}

impl<T: ElemType, M: Memory<T>> GroupedQueryAttention<T, M> {
    pub fn new(
        norm: Tensor<T, M>,
        weights: AttentionWeights<T, M>,
        head_size: u32,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rms_norm_epsilon: T,
        rope_theta: T,
    ) -> Self {
        let rms_norm = RMSNorm::new(norm, rms_norm_epsilon);
        Self {
            rms_norm,
            qb: weights.q_bias,
            qw: weights.q_weight.transpose(),
            kb: weights.k_bias,
            kw: weights.k_weight.transpose(),
            vb: weights.v_bias,
            vw: weights.v_weight.transpose(),
            ow: weights.o_weight.transpose(),
            head_size,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
        }
    }

    pub fn execute<
        MO: MemoryOwn<T>,
        D: MemoryMut<T, Base = M::Base>,
        T0: MemoryMut<T, Base = M::Base>,
        T1: MemoryMut<T, Base = M::Base>,
        T2: MemoryMut<T, Base = M::Base>,
    >(
        &self,
        kv_cache: &mut KVCache<T, MO>,
        t: u32,
        target_t_x_h: &mut Tensor<T, D>,
        tmp_2_x_d: &mut Tensor<T, T0>,
        tmp_t_x_h: &mut Tensor<T, T1>,
        tmp_t_x_nt: &mut Tensor<T, T2>,
    ) -> Result<(), MLTError>
    where
        M: Memory<T, Base = MO>,
    {
        // TODO: safety: t>0.
        let d = self.head_size;
        let h = d * self.num_attention_heads as u32;
        let n = kv_cache.n();
        let nt = n + t;
        let kvd = d * self.num_key_value_heads as u32;

        let qtmp_t_x_h = tmp_t_x_h;
        let (mut ktmp_t_x_kvd, mut vtmp_t_x_kvd) = kv_cache.allocate(t)?;
        let (mut ropetmp0_1_x_d, mut ropetmp1_1_x_d) = tmp_2_x_d.split_row(1)?;

        self.rms_norm.execute(target_t_x_h)?;

        qtmp_t_x_h.matmul(target_t_x_h, &self.qw)?;
        qtmp_t_x_h.add_assign(&self.qb)?;
        ktmp_t_x_kvd.matmul(target_t_x_h, &self.kw)?;
        ktmp_t_x_kvd.add_assign(&self.kb)?;
        vtmp_t_x_kvd.matmul(target_t_x_h, &self.vw)?;
        vtmp_t_x_kvd.add_assign(&self.vb)?;

        for i in 0..t {
            let rope = RoPE::new(&mut ropetmp0_1_x_d, n + i, self.rope_theta, d)?;

            let mut qtarget_1_x_h = qtmp_t_x_h.slice_mut(i..i + 1, 0..h);
            let mut ktarget_1_x_h = ktmp_t_x_kvd.slice_mut(i..i + 1, 0..kvd);

            rope.execute(
                &mut qtarget_1_x_h,
                &mut ropetmp1_1_x_d,
                self.num_attention_heads,
            )?;
            rope.execute(
                &mut ktarget_1_x_h,
                &mut ropetmp1_1_x_d,
                self.num_key_value_heads,
            )?;
        }

        let score_t_x_nt = tmp_t_x_nt;

        let (kref_nt_x_kvd, vref_nt_x_kvd) = kv_cache.get_kv();
        let kref_kvd_x_nt = kref_nt_x_kvd.transpose();
        for i in 0..self.num_attention_heads as u32 {
            let kvi = i / (self.num_attention_heads / self.num_key_value_heads) as u32;

            let mut qi_t_x_d = qtmp_t_x_h.slice_mut(0..t, i * d..(i + 1) * d);
            let ki_d_x_nt = kref_kvd_x_nt.slice(kvi * d..(kvi + 1) * d, 0..nt);
            let vi_nt_x_d = vref_nt_x_kvd.slice(0..nt, kvi * d..(kvi + 1) * d);

            score_t_x_nt.matmul(&qi_t_x_d, &ki_d_x_nt)?;
            score_t_x_nt.scalar_mul_assign(T::from_u32(d).inv_sqrt());
            for i in 0..t - 1 {
                let n_mask = t - 1 - i;
                let mut to_be_masked = score_t_x_nt.slice_mut(i..i + 1, nt - n_mask..nt);
                to_be_masked.fill(T::MIN);
            }
            score_t_x_nt.safe_softmax();
            qi_t_x_d.matmul(score_t_x_nt, &vi_nt_x_d)?;
        }

        target_t_x_h.matmul(qtmp_t_x_h, &self.ow)?;

        Ok(())
    }
}
