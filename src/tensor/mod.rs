use crate::device::{Device, DeviceOps, MutableDevice, OwnedDevice};
use std::marker::PhantomData;
use std::ops::Range;

pub(crate) trait ElemType: Send + Sync {
    const BYTES: usize;
}
pub struct F32;
impl ElemType for F32 {
    const BYTES: usize = 4;
}
pub struct BF16;
impl ElemType for BF16 {
    const BYTES: usize = 2;
}

pub(crate) struct Layout {
    pub(crate) offset: usize,
    pub(crate) nrow: u32,
    pub(crate) ncol: u32,
    pub(crate) stride: u32,
}

impl Layout {
    pub(crate) fn is_packed(&self) -> bool {
        self.ncol == self.stride
    }

    pub(crate) fn offset<E: ElemType>(&self, r: u32, c: u32) -> usize {
        (r * self.stride + c) as usize * E::BYTES + self.offset
    }

    fn sliced<E: ElemType>(&self, rows: Range<u32>, cols: Range<u32>) -> Layout {
        let rs = rows.start.min(self.nrow);
        let re = rows.end.max(rows.start).min(self.nrow);
        let cs = cols.start.min(self.ncol);
        let ce = cols.end.max(cols.start).min(self.ncol);
        let slice_offset = self.offset::<E>(rs, cs);

        Layout {
            offset: slice_offset,
            nrow: re - rs,
            ncol: ce - cs,
            stride: self.stride,
        }
    }
}

pub(crate) struct Tensor<E: ElemType, D: Device> {
    device: D,
    layout: Layout,
    _phantom: PhantomData<E>,
}

pub(crate) struct Trans<'a, E: ElemType, D: Device> {
    tensor: &'a Tensor<E, D>,
}

impl<E: ElemType, D: Device> Tensor<E, D> {
    pub(crate) fn new(
        device: D,
        offset: usize,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<Self, crate::Error> {
        let layout = Layout {
            offset,
            nrow,
            ncol,
            stride,
        };
        validate_space::<E>(device.len(), &layout)?;
        Ok(Self {
            device,
            layout,
            _phantom: PhantomData,
        })
    }

    pub(crate) fn transpose(&self) -> Trans<'_, E, D> {
        Trans { tensor: self }
    }

    pub(crate) fn slice(&self, rows: Range<u32>, cols: Range<u32>) -> Tensor<E, &D::Base> {
        let device = self.device.as_base();
        let layout = self.layout.sliced::<E>(rows, cols);
        Tensor {
            device,
            layout,
            _phantom: PhantomData,
        }
    }
}

impl<E: ElemType, MD: MutableDevice> Tensor<E, MD> {
    pub(crate) fn slice_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Tensor<E, &mut MD::Base> {
        let device = self.device.as_mut_base();
        let layout = self.layout.sliced::<E>(rows, cols);
        Tensor {
            device,
            layout,
            _phantom: PhantomData,
        }
    }
}

impl<E: ElemType, OD: OwnedDevice> Tensor<E, OD> {
    pub(crate) fn reshape(
        &mut self,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<(), crate::Error> {
        self.layout.nrow = nrow;
        self.layout.ncol = ncol;
        self.layout.stride = stride;

        let required = self.layout.offset::<E>(nrow - 1, ncol);
        if self.device.len() < required {
            self.device.resize(required)?;
        }

        Ok(())
    }
}

impl<E: ElemType, OD: OwnedDevice> Tensor<E, &mut OD> {
    pub(crate) fn split_row(self, mid: u32) -> Result<(Self, Self), crate::Error> {
        let Self {
            device,
            layout,
            _phantom,
        } = self;

        if mid > layout.nrow {
            return Err(crate::Error::out_of_bound(
                mid as usize,
                layout.nrow as usize,
            ));
        }

        let device_ptr = device as *mut OD;
        let offset0 = layout.offset;
        let tensor0 = Self {
            device: unsafe { &mut *device_ptr },
            layout: Layout {
                offset: offset0,
                nrow: mid,
                ncol: layout.ncol,
                stride: layout.stride,
            },
            _phantom: PhantomData,
        };

        let offset1 = layout.offset::<E>(mid, 0);
        let tensor1 = Self {
            device: unsafe { &mut *device_ptr },
            layout: Layout {
                offset: offset1,
                nrow: layout.nrow - mid,
                ncol: layout.ncol,
                stride: layout.stride,
            },
            _phantom: PhantomData,
        };

        Ok((tensor0, tensor1))
    }

    pub(crate) fn split_col(self, mid: u32) -> Result<(Self, Self), crate::Error> {
        let Self {
            device,
            layout,
            _phantom,
        } = self;

        if mid > layout.ncol {
            return Err(crate::Error::out_of_bound(
                mid as usize,
                layout.ncol as usize,
            ));
        }

        let device_ptr = device as *mut OD;
        let offset0 = layout.offset;
        let tensor0 = Self {
            device: unsafe { &mut *device_ptr },
            layout: Layout {
                offset: offset0,
                nrow: layout.nrow,
                ncol: mid,
                stride: layout.stride,
            },
            _phantom: PhantomData,
        };

        let offset1 = layout.offset::<E>(0, mid);
        let tensor1 = Self {
            device: unsafe { &mut *device_ptr },
            layout: Layout {
                offset: offset1,
                nrow: layout.nrow,
                ncol: layout.ncol - mid,
                stride: layout.stride,
            },
            _phantom: PhantomData,
        };

        Ok((tensor0, tensor1))
    }
}

impl<E: ElemType, M: MutableDevice> Tensor<E, M>
where
    M::Base: DeviceOps<E>,
{
    pub(crate) fn add<D0: Device<Base = M::Base>>(
        &mut self,
        other: &Tensor<E, D0>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        unsafe { M::Base::add(&mut self.device, &self.layout, &other.device, &other.layout) };

        Ok(())
    }

    pub(crate) fn argmax(&self) -> Result<u32, crate::Error> {
        validate_shape(self.layout.nrow, self.layout.ncol, 1, self.layout.ncol)?;

        Ok(unsafe { M::Base::argmax(&self.device, &self.layout) })
    }

    pub(crate) fn cast_from_bf16<D0: Device<Base = M::Base>>(
        &mut self,
        other: &Tensor<BF16, D0>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        unsafe {
            M::Base::cast_from_bf16(&mut self.device, &self.layout, &other.device, &other.layout)
        };

        Ok(())
    }

    pub(crate) fn copy<D0: Device<Base = M::Base>>(
        &mut self,
        other: &Tensor<E, D0>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        unsafe { M::Base::copy(&mut self.device, &self.layout, &other.device, &other.layout) };

        Ok(())
    }

    // self[i] = a[i]*b[i]*alpha
    pub(crate) fn mul_elementwise<D0: Device<Base = M::Base>, D1: Device<Base = M::Base>>(
        &mut self,
        a: &Tensor<E, D0>,
        b: &Tensor<E, D1>,
        alpha: f32,
    ) -> Result<(), crate::Error> {
        validate_shape(
            a.layout.nrow,
            a.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;
        validate_shape(
            b.layout.nrow,
            b.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        unsafe {
            M::Base::mul_elementwise(
                &mut self.device,
                &self.layout,
                &a.device,
                &a.layout,
                &b.device,
                &b.layout,
                alpha,
            )
        };

        Ok(())
    }

    // self = AB
    // shape of self must be (A.nrow * B.ncol)
    // B.nrow must be equal to A.ncol
    pub(crate) fn mul<D0: Device<Base = M::Base>, D1: Device<Base = M::Base>>(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<E, D0>,
        #[allow(non_snake_case)] B: &Tensor<E, D1>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            self.layout.nrow,
            self.layout.ncol,
            A.layout.nrow,
            B.layout.ncol,
        )?;
        validate_shape(
            A.layout.nrow,
            A.layout.ncol,
            self.layout.nrow,
            B.layout.nrow,
        )?;

        unsafe {
            M::Base::mul_mn_mk_kn(
                &mut self.device,
                &self.layout,
                &A.device,
                &A.layout,
                &B.device,
                &B.layout,
            )
        };

        Ok(())
    }

    // self = AB^T
    // shape of self must be (A.nrow * B.ncol)
    // B.nrow must be equal to A.ncol
    pub(crate) fn mul_bt<D0: Device<Base = M::Base>, D1: Device<Base = M::Base>>(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<E, D0>,
        #[allow(non_snake_case)] Bt: &Trans<E, D1>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            self.layout.nrow,
            self.layout.ncol,
            A.layout.nrow,
            Bt.tensor.layout.nrow,
        )?;
        validate_shape(
            A.layout.nrow,
            A.layout.ncol,
            self.layout.nrow,
            Bt.tensor.layout.ncol,
        )?;

        unsafe {
            M::Base::mul_mn_mk_knt(
                &mut self.device,
                &self.layout,
                &A.device,
                &A.layout,
                &Bt.tensor.device,
                &Bt.tensor.layout,
            )
        };

        Ok(())
    }

    // self = AB + c
    // shape of self must be (A.nrow * B.ncol)
    // A.ncol may 1536
    // B.nrow must be equal to A.ncol
    // shape of C must be (1 * B.ncol)
    // c is expanded in calculation
    pub(crate) fn muladd_broadcast<
        D0: Device<Base = M::Base>,
        D1: Device<Base = M::Base>,
        D2: Device<Base = M::Base>,
    >(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<E, D0>,
        #[allow(non_snake_case)] B: &Tensor<E, D1>,
        c: &Tensor<E, D2>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            self.layout.nrow,
            self.layout.ncol,
            A.layout.nrow,
            B.layout.ncol,
        )?;
        validate_shape(
            A.layout.nrow,
            A.layout.ncol,
            self.layout.nrow,
            B.layout.nrow,
        )?;
        validate_shape(
            B.layout.nrow,
            B.layout.ncol,
            A.layout.ncol,
            self.layout.ncol,
        )?;
        validate_shape(c.layout.nrow, c.layout.ncol, 1, self.layout.ncol)?;

        unsafe {
            M::Base::mul_mn_mk_kn_1n(
                &mut self.device,
                &self.layout,
                &A.device,
                &A.layout,
                &B.device,
                &B.layout,
                &c.device,
                &c.layout,
            )
        };

        Ok(())
    }

    // self = AB^T + c
    // shape of self must be (A.nrow * B.ncol)
    // A.ncol may 1536
    // B.nrow must be equal to A.ncol
    // shape of C must be (1 * B.ncol)
    // c is expanded in calculation
    pub(crate) fn muladd_bt_broadcast<
        D0: Device<Base = M::Base>,
        D1: Device<Base = M::Base>,
        D2: Device<Base = M::Base>,
    >(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<E, D0>,
        #[allow(non_snake_case)] B: &Trans<'_, E, D1>,
        c: &Tensor<E, D2>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            self.layout.nrow,
            self.layout.ncol,
            A.layout.nrow,
            B.tensor.layout.nrow,
        )?;
        validate_shape(
            A.layout.nrow,
            A.layout.ncol,
            self.layout.nrow,
            B.tensor.layout.ncol,
        )?;
        validate_shape(
            B.tensor.layout.ncol,
            B.tensor.layout.nrow,
            A.layout.ncol,
            self.layout.ncol,
        )?;
        validate_shape(c.layout.nrow, c.layout.ncol, 1, self.layout.ncol)?;

        unsafe {
            M::Base::mul_mn_mk_knt_1n(
                &mut self.device,
                &self.layout,
                &A.device,
                &A.layout,
                &B.tensor.device,
                &B.tensor.layout,
                &c.device,
                &c.layout,
            )
        };

        Ok(())
    }

    pub(crate) fn rms_norm<D0: Device<Base = M::Base>>(
        &mut self,
        weight: &Tensor<E, D0>,
        epsilon: f32,
    ) -> Result<(), crate::Error> {
        validate_shape(weight.layout.nrow, weight.layout.ncol, 1, self.layout.ncol)?;

        unsafe {
            M::Base::rms_norm(
                &mut self.device,
                &self.layout,
                &weight.device,
                &weight.layout,
                epsilon,
            )
        };
        Ok(())
    }

    pub(crate) fn rope_vector(
        &mut self,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<(), crate::Error> {
        validate_shape(self.layout.nrow, self.layout.ncol, 1, head_size)?;

        let dst = self.slice_mut(0..1, 0..head_size);
        let (mut dst0, mut dst1) = dst.split_col(head_size / 2)?;

        let k = token_index as f32;
        let d = head_size as f32;
        unsafe {
            M::Base::rope_cos(&mut dst0.device, &dst0.layout, k, rope_theta, d);
            M::Base::rope_sin(&mut dst1.device, &dst1.layout, k, rope_theta, d);
        }

        Ok(())
    }

    // SiLU : x / (1 + e^(-x)) (element op)
    pub(crate) fn silu(&mut self) -> () {
        unsafe { M::Base::silu(&mut self.device, &self.layout) };
    }

    // self = safe_softmax(alpha * self). alpha > 0
    pub(crate) fn safe_softmax(&mut self, alpha: f32) -> () {
        unsafe { M::Base::safe_softmax(&mut self.device, &self.layout, alpha) };
    }
}

fn validate_space<E: ElemType>(memory_len: usize, layout: &Layout) -> Result<(), crate::Error> {
    if layout.nrow == 0 || layout.ncol == 0 {
        return Ok(());
    }
    let required = layout.offset::<E>(layout.nrow - 1, layout.ncol);
    if memory_len < required {
        return Err(crate::Error::insufficient_storage_space(
            required, memory_len,
        ));
    }
    Ok(())
}

fn validate_shape(
    nrow_actual: u32,
    ncol_actual: u32,
    nrow_expected: u32,
    ncol_expected: u32,
) -> Result<(), crate::Error> {
    if nrow_expected != nrow_actual {
        return Err(crate::Error::shape_mismatch(
            nrow_expected as usize,
            nrow_actual as usize,
        ));
    }
    if ncol_expected != ncol_actual {
        return Err(crate::Error::shape_mismatch(
            ncol_expected as usize,
            ncol_actual as usize,
        ));
    }
    Ok(())
}

#[cfg(test)]
impl<D: Device<Base = crate::device::Cpu>> Tensor<F32, D> {
    /// Test Helper Function
    pub(crate) fn assert<const N: usize>(&self, answer: &[[f32; N]]) -> () {
        assert!(
            self.layout.nrow as usize == answer.len() && self.layout.ncol as usize == N,
            "invalid test storage"
        );

        let mut ptr = unsafe { self.device.as_ptr().byte_add(self.layout.offset) } as *const f32;
        for i in 0..self.layout.nrow as usize {
            for j in 0..self.layout.ncol as usize {
                let expected = answer[i][j];
                let actual = unsafe { *(ptr.add(j)) };
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "storage[{}][{}] mismatch: actual {}, expected {}",
                    i,
                    j,
                    actual,
                    expected,
                );
            }
            ptr = unsafe { ptr.add(self.layout.stride as usize) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::device::{Cpu, MutableDevice, OwnedDevice};

    fn device_from_bf16(src: &[u16]) -> Cpu {
        let len = src.len() * size_of::<u16>();
        let mut device = Cpu::new(len).expect("creating mmap should succeed");
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), device.as_mut_ptr() as *mut u16, src.len())
        };
        device
    }

    #[test]
    fn case01_copy_subtensor() {
        let src = Tensor::new(
            Cpu::from([1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0].as_slice()),
            0,
            2,
            3,
            3,
        )
        .expect("creating tensor should succeed");

        let mut dst = Tensor::new(
            Cpu::from([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0].as_slice()),
            0,
            3,
            4,
            4,
        )
        .expect("creating tensor should succeed");

        dst.slice_mut(1..3, 1..4)
            .copy(&src)
            .expect("`copy` should succeed");

        dst.assert(&[
            [9.0, 9.0, 9.0, 9.0],
            [9.0, 1.0, 2.0, 3.0],
            [9.0, 2.0, 3.0, 4.0],
        ]);
    }

    #[test]
    fn case02_cast_bf16_to_f32() {
        let src = Tensor::new(
            device_from_bf16(&[0x3f80, 0x4000, 0x4040, 0x4080]),
            0,
            2,
            2,
            2,
        )
        .expect("creating tensor should succeed");

        let mut dst = Tensor::new(Cpu::from([0.0, 0.0, 0.0, 0.0].as_slice()), 0, 2, 2, 2)
            .expect("creating tensor should succeed");

        dst.cast_from_bf16(&src).expect("`cast` should succeed");

        dst.assert(&[[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn case03_rms_norm() {
        let mut x = Tensor::new(Cpu::from([3.0, 4.0, 0.0, 5.0].as_slice()), 0, 2, 2, 2)
            .expect("creating tensor should succeed");

        let w = Tensor::new(Cpu::from([2.0, 0.5].as_slice()), 0, 1, 2, 2)
            .expect("creating tensor should succeed");

        x.rms_norm(&w, 0.0)
            .expect("applying rms_norm should succeed");

        x.assert(&[[1.6970563, 0.56568545], [0.0, 0.70710677]]);
    }
}
