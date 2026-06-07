mod cpu;

use crate::tensor::{ElemType, Layout};
pub use cpu::Cpu;

pub(crate) trait Device: Send + Sync {
    type Base: OwnedDevice;

    fn as_base(&self) -> &Self::Base;
    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const ();
}

pub(crate) trait MutableDevice: Device {
    fn as_mut_base(&mut self) -> &mut Self::Base;
    fn as_mut_ptr(&mut self) -> *mut ();
}

pub(crate) trait OwnedDevice:
    MutableDevice<Base = Self> + TryFrom<std::fs::File, Error = crate::Error>
{
    fn new(len: usize) -> Result<Self, crate::Error>;
    fn resize(&mut self, len: usize) -> Result<(), crate::Error>;
}

impl<OD: OwnedDevice> Device for &OD {
    type Base = OD::Base;

    fn len(&self) -> usize {
        (**self).len()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const () {
        (**self).as_ptr()
    }
}

impl<OD: OwnedDevice> Device for &mut OD {
    type Base = OD::Base;

    fn len(&self) -> usize {
        (**self).len()
    }
    fn as_base(&self) -> &Self::Base {
        (**self).as_base()
    }
    fn as_ptr(&self) -> *const () {
        (**self).as_ptr()
    }
}

impl<OD: OwnedDevice> MutableDevice for &mut OD {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        (**self).as_mut_base()
    }
    fn as_mut_ptr(&mut self) -> *mut () {
        (**self).as_mut_ptr()
    }
}

pub(crate) trait DeviceOps<E: ElemType>: OwnedDevice {
    unsafe fn add<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> ();
    unsafe fn argmax<D: Device<Base = Self>>(src: &D, src_layout: &Layout) -> u32;
    unsafe fn cast_from_bf16<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> ();
    unsafe fn copy<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> ();
    unsafe fn mul_elementwise<
        M: MutableDevice<Base = Self>,
        D0: Device<Base = Self>,
        D1: Device<Base = Self>,
    >(
        dst: &mut M,
        dst_layout: &Layout,
        src0: &D0,
        src0_layout: &Layout,
        src1: &D1,
        src1_layout: &Layout,
        alpha: f32,
    ) -> ();
    unsafe fn mul_mn_mk_kn<
        M: MutableDevice<Base = Self>,
        D0: Device<Base = Self>,
        D1: Device<Base = Self>,
    >(
        dst: &mut M,
        dst_layout: &Layout,
        src0: &D0,
        src0_layout: &Layout,
        src1: &D1,
        src1_layout: &Layout,
    ) -> ();
    unsafe fn mul_mn_mk_knt<
        M: MutableDevice<Base = Self>,
        D0: Device<Base = Self>,
        D1: Device<Base = Self>,
    >(
        dst: &mut M,
        dst_layout: &Layout,
        src0: &D0,
        src0_layout: &Layout,
        src1: &D1,
        src1_layout_t: &Layout,
    ) -> ();
    unsafe fn mul_mn_mk_kn_1n<
        M: MutableDevice<Base = Self>,
        D0: Device<Base = Self>,
        D1: Device<Base = Self>,
        D2: Device<Base = Self>,
    >(
        dst: &mut M,
        dst_layout: &Layout,
        src0: &D0,
        src0_layout: &Layout,
        src1: &D1,
        src1_layout: &Layout,
        src2: &D2,
        src2_layout: &Layout,
    ) -> ();
    unsafe fn rms_norm<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
        epsilon: f32,
    ) -> ();
    unsafe fn rope_cos<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        k: f32,
        theta: f32,
        d: f32,
    ) -> ();
    unsafe fn rope_sin<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        k: f32,
        theta: f32,
        d: f32,
    ) -> ();
    unsafe fn silu<M: MutableDevice<Base = Self>>(dst: &mut M, dst_layout: &Layout) -> ();
    unsafe fn softmax<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        alpha: f32,
    ) -> ();
}
