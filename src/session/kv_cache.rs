use crate::device::OwnedDevice;
use crate::tensor::{ElemType, Tensor};

pub(crate) struct KVCache<E: ElemType, OD: OwnedDevice> {
    k: Tensor<E, OD>,
    v: Tensor<E, OD>,

    kv_head_size: u32,
    n: u32,
}

impl<E: ElemType, OD: OwnedDevice> KVCache<E, OD> {
    pub(crate) fn new(
        k_device: OD,
        v_device: OD,
        head_size: u32,
        num_key_value_heads: usize,
        n: u32,
    ) -> Result<Self, crate::Error> {
        let kv_head_size = head_size * num_key_value_heads as u32;
        let kvd = kv_head_size;

        let k = Tensor::new(k_device, 0, n, kvd, kvd)?;
        let v = Tensor::new(v_device, 0, n, kvd, kvd)?;
        Ok(Self {
            k,
            v,
            kv_head_size,
            n,
        })
    }

    pub(crate) fn n(&self) -> u32 {
        self.n
    }

    pub(crate) fn allocate(
        &mut self,
    ) -> Result<(Tensor<E, &mut OD::Base>, Tensor<E, &mut OD::Base>), crate::Error> {
        let kvd = self.kv_head_size;
        let n = self.n;

        self.k.reshape(n + 1, kvd, kvd)?;
        self.v.reshape(n + 1, kvd, kvd)?;

        let k_mut = self.k.slice_mut(n..n + 1, 0..kvd);
        let v_mut = self.v.slice_mut(n..n + 1, 0..kvd);

        self.n += 1;

        Ok((k_mut, v_mut))
    }

    pub(crate) fn get_kv(&self) -> (Tensor<E, &OD::Base>, Tensor<E, &OD::Base>) {
        let kvd = self.kv_head_size;

        let k_ref = self.k.slice(0..self.n, 0..kvd);
        let v_ref = self.v.slice(0..self.n, 0..kvd);

        (k_ref, v_ref)
    }
}
