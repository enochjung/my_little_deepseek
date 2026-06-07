use crate::device::OwnedDevice;
use crate::tensor::{ElemType, Tensor};

pub(crate) struct KVCache<E: ElemType, OD: OwnedDevice> {
    k: Tensor<E, OD>,
    v: Tensor<E, OD>,

    head_size: u32,
    n: u32,
}

impl<E: ElemType, OD: OwnedDevice> KVCache<E, OD> {
    pub(crate) fn new(
        k_device: OD,
        v_device: OD,
        head_size: u32,
        n: u32,
    ) -> Result<Self, crate::Error> {
        let k = Tensor::new(k_device, 0, n, head_size, head_size)?;
        let v = Tensor::new(v_device, 0, n, head_size, head_size)?;
        Ok(Self { k, v, head_size, n })
    }

    pub(crate) fn n(&self) -> u32 {
        self.n
    }

    pub(crate) fn allocate(
        &mut self,
    ) -> Result<(Tensor<E, &mut OD::Base>, Tensor<E, &mut OD::Base>), crate::Error> {
        let d = self.head_size;
        let n = self.n;

        self.k.reshape(n + 1, d, d)?;
        self.v.reshape(n + 1, d, d)?;

        let k_mut = self.k.slice_mut(n..n + 1, 0..d);
        let v_mut = self.v.slice_mut(n..n + 1, 0..d);

        Ok((k_mut, v_mut))
    }

    pub(crate) fn get_kv(&self) -> (Tensor<E, &OD::Base>, Tensor<E, &OD::Base>) {
        let k_ref = self.k.slice(0..self.n, 0..self.head_size);
        let v_ref = self.v.slice(0..self.n, 0..self.head_size);

        (k_ref, v_ref)
    }
}
