use crate::storage::*;
use crate::tensor::*;
use std::marker::PhantomData;

pub(crate) struct KVCache<E: ElemType, MO: Mutable + Owned> {
    // todo
    _phantom: PhantomData<(E, MO)>,
}

impl<E: ElemType, MO: Mutable + Owned> KVCache<E, MO> {
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub(crate) fn len(&self) -> usize {
        todo!()
    }

    pub(crate) fn allocate(
        &mut self,
    ) -> Result<(Tensor<E, &mut MO>, Tensor<E, &mut MO>), crate::Error> {
        todo!()
    }

    pub(crate) fn get_kv(&self) -> (Tensor<E, &MO>, Tensor<E, &MO>) {
        todo!()
    }
}
