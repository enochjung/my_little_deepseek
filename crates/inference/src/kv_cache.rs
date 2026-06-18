use core::{ElemType, MLTError, MatrixLayout, MemoryOwn};

use crate::tensor::Tensor;

pub struct KVCache<T: ElemType, MO: MemoryOwn<T>> {
    k: Tensor<T, MO>,
    v: Tensor<T, MO>,

    kv_head_size: u32,
    n: u32,
}

impl<T: ElemType, MO: MemoryOwn<T>> KVCache<T, MO> {
    pub fn new(
        k_mem: MO,
        v_mem: MO,
        head_size: u32,
        num_key_value_heads: usize,
        n: u32,
    ) -> Result<Self, MLTError> {
        let kv_head_size = head_size * num_key_value_heads as u32;
        let kvd = kv_head_size;

        let k = Tensor::new(k_mem, MatrixLayout::new(0, n, kvd, kvd, 1))?;
        let v = Tensor::new(v_mem, MatrixLayout::new(0, n, kvd, kvd, 1))?;
        Ok(Self {
            k,
            v,
            kv_head_size,
            n,
        })
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    pub fn allocate(
        &mut self,
        t: u32,
    ) -> Result<(Tensor<T, &mut MO>, Tensor<T, &mut MO>), MLTError> {
        let kvd = self.kv_head_size;
        let n = self.n;
        let nt = n + t;

        self.k.reshape(nt, kvd, kvd, 1)?;
        self.v.reshape(nt, kvd, kvd, 1)?;

        let k_mut = self.k.slice_mut(n..nt, 0..kvd);
        let v_mut = self.v.slice_mut(n..nt, 0..kvd);

        self.n = nt;

        Ok((k_mut, v_mut))
    }

    pub fn get_kv(&self) -> (Tensor<T, &MO::Base>, Tensor<T, &MO::Base>) {
        let kvd = self.kv_head_size;

        let k_ref = self.k.slice(0..self.n, 0..kvd);
        let v_ref = self.v.slice(0..self.n, 0..kvd);

        (k_ref, v_ref)
    }
}
