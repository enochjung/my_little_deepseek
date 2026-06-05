use crate::kernel;
use crate::storage::*;
use std::marker::PhantomData;
use std::ops::Range;

pub(crate) trait ElemType {
    const BYTES: usize = 4;
}
pub(crate) struct F32;
impl ElemType for F32 {
    const BYTES: usize = 4;
}
pub(crate) struct BF16;
impl ElemType for BF16 {
    const BYTES: usize = 2;
}

struct Layout {
    offset: usize,
    nrow: u32,
    ncol: u32,
    stride: u32,
    is_row_major: bool,
}

impl Layout {
    fn transpose(mut self) -> Self {
        self.is_row_major = !self.is_row_major;
        core::mem::swap(&mut self.nrow, &mut self.ncol);
        self
    }

    fn index(&self, row_idx: u32, col_idx: u32) -> u32 {
        match self.is_row_major {
            true => row_idx * self.stride + col_idx,
            false => col_idx * self.stride + row_idx,
        }
    }

    fn n_lines(&self) -> u32 {
        match self.is_row_major {
            true => self.nrow,
            false => self.ncol,
        }
    }

    fn line_elems(&self) -> u32 {
        match self.is_row_major {
            true => self.ncol,
            false => self.nrow,
        }
    }

    fn is_packed(&self) -> bool {
        self.stride == self.line_elems()
    }
}

pub(crate) struct Tensor<E: ElemType, S: Storage> {
    storage: S,
    layout: Layout,
    _phantom: PhantomData<E>,
}

impl<E: ElemType, S: Storage> Tensor<E, S> {
    pub(crate) fn new(
        storage: S,
        offset: usize,
        nrow: u32,
        ncol: u32,
        stride: u32,
        is_row_major: bool,
    ) -> Result<Self, crate::Error> {
        let layout = Layout {
            offset,
            nrow,
            ncol,
            stride,
            is_row_major,
        };
        validate_space::<E>(storage.len(), &layout)?;
        Ok(Self {
            storage,
            layout,
            _phantom: PhantomData,
        })
    }

    pub(crate) fn transpose(mut self) -> Self {
        self.layout = self.layout.transpose();
        self
    }

    fn sliced_layout(&self, rows: Range<u32>, cols: Range<u32>) -> Layout {
        let rs = rows.start.min(self.layout.nrow);
        let re = rows.end.max(rows.start).min(self.layout.nrow);
        let cs = cols.start.min(self.layout.ncol);
        let ce = cols.end.max(cols.start).min(self.layout.ncol);
        let slice_offset = self.layout.index(rs, cs) as usize * E::BYTES;

        Layout {
            offset: self.layout.offset + slice_offset,
            nrow: re - rs,
            ncol: ce - cs,
            stride: self.layout.stride,
            is_row_major: self.layout.is_row_major,
        }
    }

    pub(crate) fn slice(&self, rows: Range<u32>, cols: Range<u32>) -> Tensor<E, &S> {
        let storage = &self.storage;
        Tensor {
            storage,
            layout: self.sliced_layout(rows, cols),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn take_storage(self) -> S
    where
        S: Owned,
    {
        self.storage
    }
}

impl<E: ElemType, M: Mutable> Tensor<E, M> {
    pub(crate) fn slice_mut(&mut self, rows: Range<u32>, cols: Range<u32>) -> Tensor<E, &mut M> {
        let layout = self.sliced_layout(rows, cols);
        let storage = &mut self.storage;
        Tensor {
            storage,
            layout,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn into_readonly(self) -> Tensor<E, M::ReadOnly>
    where
        M: Owned,
    {
        let storage = self.storage.into_readonly();
        Tensor {
            storage,
            layout: self.layout,
            _phantom: PhantomData,
        }
    }
}

impl<E: ElemType, M: Mutable> Tensor<E, &mut M> {
    pub(crate) fn split_row(self, mid: u32) -> Result<(Self, Self), crate::Error> {
        let Self {
            storage,
            layout,
            _phantom,
        } = self;

        if mid > layout.nrow {
            return Err(crate::Error::out_of_bound(
                mid as usize,
                layout.nrow as usize,
            ));
        }

        let storage_ptr = storage as *mut M;
        let offset0 = layout.offset;
        let tensor0 = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout {
                offset: offset0,
                nrow: mid,
                ncol: layout.ncol,
                stride: layout.stride,
                is_row_major: layout.is_row_major,
            },
            _phantom: PhantomData,
        };

        let offset1 = if layout.is_row_major {
            mid * layout.stride
        } else {
            mid
        } as usize
            * E::BYTES
            + layout.offset;
        let tensor1 = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout {
                offset: offset1,
                nrow: layout.nrow - mid,
                ncol: layout.ncol,
                stride: layout.stride,
                is_row_major: layout.is_row_major,
            },
            _phantom: PhantomData,
        };

        Ok((tensor0, tensor1))
    }

    pub(crate) fn split_col(self, mid: u32) -> Result<(Self, Self), crate::Error> {
        let Self {
            storage,
            layout,
            _phantom,
        } = self;

        if mid > layout.ncol {
            return Err(crate::Error::out_of_bound(
                mid as usize,
                layout.ncol as usize,
            ));
        }

        let storage_ptr = storage as *mut M;
        let offset0 = layout.offset;
        let tensor0 = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout {
                offset: offset0,
                nrow: layout.nrow,
                ncol: mid,
                stride: layout.stride,
                is_row_major: layout.is_row_major,
            },
            _phantom: PhantomData,
        };

        let offset1 = if layout.is_row_major {
            mid
        } else {
            mid * layout.stride
        } as usize
            * E::BYTES
            + layout.offset;
        let tensor1 = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout {
                offset: offset1,
                nrow: layout.nrow,
                ncol: layout.ncol - mid,
                stride: layout.stride,
                is_row_major: layout.is_row_major,
            },
            _phantom: PhantomData,
        };

        Ok((tensor0, tensor1))
    }
}

impl<MH: Mutable<Loc = Host>> Tensor<F32, MH> {
    pub(crate) fn copy<S: Storage<Loc = Host>>(
        &mut self,
        other: &Tensor<F32, S>,
    ) -> Result<(), crate::Error> {
        if self.layout.is_row_major != other.layout.is_row_major {
            return Err(crate::Error::operation_not_supported(
                "`copy` with different layout majority",
            ));
        }
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        let dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) };
        let src = unsafe { other.storage.as_ptr().byte_add(other.layout.offset) };

        if self.layout.is_packed() && other.layout.is_packed() {
            let len = self.layout.nrow as usize * self.layout.ncol as usize * F32::BYTES;
            unsafe { kernel::copy(dst, src, len) };
        } else {
            let n_lines = self.layout.n_lines();
            let len = self.layout.line_elems() as usize * F32::BYTES;
            let dst_stride_bytes = self.layout.stride as usize * F32::BYTES;
            let src_stride_bytes = other.layout.stride as usize * F32::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..n_lines {
                unsafe { kernel::copy(dst, src, len) };
                dst = unsafe { dst.byte_add(dst_stride_bytes) };
                src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }

        Ok(())
    }

    pub(crate) fn cast<S: Storage<Loc = Host>>(
        &mut self,
        other: &Tensor<BF16, S>,
    ) -> Result<(), crate::Error> {
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        let dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) } as *mut f32;
        let src = unsafe { other.storage.as_ptr().byte_add(other.layout.offset) };
        if self.layout.is_packed() && other.layout.is_packed() {
            let n = self.layout.nrow as usize * self.layout.ncol as usize;
            unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
        } else {
            let n_lines = self.layout.n_lines();
            let n = self.layout.line_elems() as usize;
            let dst_stride_bytes = self.layout.stride as usize * F32::BYTES;
            let src_stride_bytes = other.layout.stride as usize * BF16::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..n_lines {
                unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
                dst = unsafe { dst.byte_add(dst_stride_bytes) };
                src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }
        Ok(())
    }

    pub(crate) fn rope_cs(
        &mut self,
        token_index: u32,
        rope_theta: f32,
        head_size: u32,
    ) -> Result<(), crate::Error> {
        validate_shape(self.layout.nrow, self.layout.ncol, 1, head_size)?;

        let x = self.storage.as_mut_ptr() as *mut f32;
        let n = head_size as usize / 2;
        let k = token_index as f32;
        let d = head_size as f32;

        unsafe { kernel::rope_cos_n(x, n, k, rope_theta, d) };
        unsafe { kernel::rope_sin_n(x.add(n), n, k, rope_theta, d) };

        Ok(())
    }

    pub(crate) fn add<S: Storage<Loc = Host>>(
        &mut self,
        other: &Tensor<F32, S>,
    ) -> Result<(), crate::Error> {
        if self.layout.is_row_major != other.layout.is_row_major {
            return Err(crate::Error::operation_not_supported(
                "`add` with mismatched tensor layout majority",
            ));
        }
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        let dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) } as *mut f32;
        let src = unsafe { other.storage.as_ptr().byte_add(other.layout.offset) } as *const f32;
        if self.layout.is_packed() && other.layout.is_packed() {
            let n = self.layout.nrow as usize * self.layout.ncol as usize;
            unsafe { kernel::add_n_n(dst, src, n) };
        } else {
            let n_lines = self.layout.n_lines();
            let n = self.layout.line_elems() as usize;
            let dst_stride_bytes = self.layout.stride as usize * F32::BYTES;
            let src_stride_bytes = other.layout.stride as usize * F32::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..n_lines {
                unsafe { kernel::add_n_n(dst, src, n) };
                dst = unsafe { dst.byte_add(dst_stride_bytes) };
                src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }

        Ok(())
    }

    pub(crate) fn mul_elementwise<S0: Storage<Loc = Host>, S1: Storage<Loc = Host>>(
        &mut self,
        a: &Tensor<F32, S0>,
        b: &Tensor<F32, S1>,
        alpha: f32,
    ) -> Result<(), crate::Error> {
        todo!()
        /*
        if self.layout.is_row_major != other.layout.is_row_major {
            return Err(crate::Error::operation_not_supported(
                "`add` with mismatched tensor layout majority",
            ));
        }
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        let is_other_packed = other.layout.line_elems() == other.layout.stride;
        let y = self.storage.as_mut_ptr() as *mut f32;
        let x = unsafe { other.storage.as_ptr().byte_add(other.layout.offset) } as *const f32;

        if is_other_packed {
            let n = self.layout.nrow as usize * self.layout.ncol as usize;
            unsafe { kernel::add_n_n(y as *mut f32, x as *const f32, n) };
        } else {
            let mut y = y;
            let mut x = x;

            let n_lines = self.layout.n_lines();
            let line_elems = self.layout.line_elems() as usize;

            for _ in 0..n_lines {
                unsafe { kernel::add_n_n(y, x, line_elems) };
                y = unsafe { y.add(self.layout.stride as usize) };
                x = unsafe { x.add(other.layout.stride as usize) };
            }
        }

        Ok(())
        */
    }

    pub(crate) fn rms_norm<S: Storage<Loc = Host>>(
        &mut self,
        weight: &Tensor<F32, S>,
        epsilon: f32,
    ) -> Result<(), crate::Error> {
        if self.layout.is_row_major != true {
            return Err(crate::Error::operation_not_supported(
                "`rms_norm` with column-major",
            ));
        }
        validate_shape(weight.layout.nrow, weight.layout.ncol, 1, self.layout.ncol)?;

        let n_lines = self.layout.n_lines();
        let n = self.layout.line_elems() as usize;
        let dst_stride_bytes = self.layout.stride as usize * F32::BYTES;

        let mut dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) } as *mut f32;
        let src = unsafe { weight.storage.as_ptr().byte_add(weight.layout.offset) } as *const f32;
        for _ in 0..n_lines {
            let rms = unsafe { kernel::rms_n(dst as *const f32, n) };
            let scale = 1.0 / (rms + epsilon);
            unsafe { kernel::mul_n_n(dst, src, scale, n) };
            dst = unsafe { dst.byte_add(dst_stride_bytes) };
        }
        Ok(())
    }

    pub(crate) fn softmax(&mut self, alpha: f32) -> Result<(), crate::Error> {
        todo!()
    }

    /*
    pub(crate) fn silu(&mut self) -> Result<(), crate::Error> {
        todo!()
        let stride = self.layout().stride;
        let n_lines = self.layout().n_lines();
        let line_elems = self.layout().line_elems();
        let is_packed = self.layout().is_packed();

        if is_packed {
            let n = n_lines as usize * line_elems as usize;
            unsafe {
                let x = self.as_mut_ptr(0, 0)? as *mut f32;
                x86_64::silu_n(x, n);
            }
        } else {
            let n = line_elems as usize;
            unsafe {
                let mut x = self.as_mut_ptr(0, 0)? as *mut f32;
                for _ in 0..n_lines {
                    x86_64::silu_n(x, n);
                    x = x.add(stride as usize);
                }
            }
        }

        Ok(())
    }
     */

    // self = AB
    // shape of self must be (A.nrow * B.ncol)
    // B.nrow must be equal to A.ncol
    pub(crate) fn mul<S0: Storage<Loc = Host>, S1: Storage<Loc = Host>>(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<F32, S0>,
        #[allow(non_snake_case)] B: &Tensor<F32, S1>,
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
            self.layout.ncol,
            B.layout.nrow,
        )?;

        let m = A.layout.nrow as usize;
        let k = A.layout.ncol as usize;
        let n = B.layout.ncol as usize;

        let dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) } as *mut f32;
        let a = unsafe { A.storage.as_ptr().byte_add(A.layout.offset) } as *const f32;
        let b = unsafe { B.storage.as_ptr().byte_add(B.layout.offset) } as *const f32;

        unsafe {
            kernel::mul_mk_kn(
                dst,
                self.layout.is_row_major,
                self.layout.stride as usize,
                a,
                A.layout.is_row_major,
                A.layout.stride as usize,
                b,
                B.layout.is_row_major,
                B.layout.stride as usize,
                m,
                k,
                n,
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
        S0: Storage<Loc = Host>,
        S1: Storage<Loc = Host>,
        S2: Storage<Loc = Host>,
    >(
        &mut self,
        #[allow(non_snake_case)] A: &Tensor<F32, S0>,
        #[allow(non_snake_case)] B: &Tensor<F32, S1>,
        c: &Tensor<F32, S2>,
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
            self.layout.ncol,
            B.layout.nrow,
        )?;
        validate_shape(c.layout.nrow, c.layout.ncol, 1, B.layout.ncol)?;

        let m = A.layout.nrow as usize;
        let k = A.layout.ncol as usize;
        let n = B.layout.ncol as usize;

        let dst = unsafe { self.storage.as_mut_ptr().byte_add(self.layout.offset) } as *mut f32;
        let a = unsafe { A.storage.as_ptr().byte_add(A.layout.offset) } as *const f32;
        let b = unsafe { B.storage.as_ptr().byte_add(B.layout.offset) } as *const f32;
        let c = unsafe { c.storage.as_ptr().byte_add(c.layout.offset) } as *const f32;

        // TODO: validate size
        let mut buf: Vec<f32> = vec![0.0f32; n as usize];
        let buf_ptr = buf.as_mut_ptr();

        unsafe {
            kernel::muladd_mk_kn_1n(
                dst,
                self.layout.is_row_major,
                self.layout.stride as usize,
                a,
                A.layout.is_row_major,
                A.layout.stride as usize,
                b,
                B.layout.is_row_major,
                B.layout.stride as usize,
                c,
                buf_ptr,
                m,
                k,
                n,
            )
        };

        Ok(())
    }
}

fn validate_space<E: ElemType>(storage_len: usize, layout: &Layout) -> Result<(), crate::Error> {
    if layout.n_lines() == 0 {
        return Ok(());
    }
    let required =
        layout.index(layout.n_lines() - 1, layout.line_elems()) as usize * E::BYTES + layout.offset;
    if storage_len < required {
        return Err(crate::Error::insufficient_storage_space(
            required,
            storage_len,
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
impl<S: Storage<Loc = Host>> Tensor<F32, S> {
    /// Test Helper Function
    pub(crate) fn assert<const N: usize>(&self, answer: &[[f32; N]]) -> () {
        let n_lines = self.layout.n_lines() as usize;
        let line_elems = self.layout.line_elems() as usize;

        assert!(n_lines == answer.len(), "invalid test storage");
        assert!(line_elems == N, "invalid test storage");

        let mut ptr = unsafe { self.storage.as_ptr().byte_add(self.layout.offset) } as *const f32;
        for i in 0..n_lines {
            for j in 0..line_elems {
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
    use super::*;

    fn mmap_from_bf16(value: &[u16]) -> Mmap {
        let len = value.len() * BF16::BYTES;
        let mut mmap = MmapMut::new(len).expect("creating mmap should succeed");
        unsafe {
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                mmap.as_mut_ptr() as *mut u16,
                value.len(),
            )
        };
        mmap.into_readonly()
    }

    #[test]
    fn case01_copy_subtensor() {
        let src = Tensor::<F32, _>::new(
            MmapMut::from([1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0].as_slice()),
            0,
            2,
            3,
            3,
            true,
        )
        .expect("creating tensor should succeed")
        .into_readonly();

        let mut dst = Tensor::<F32, _>::new(
            MmapMut::from([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0].as_slice()),
            0,
            3,
            4,
            4,
            true,
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
        let src = Tensor::<BF16, _>::new(
            mmap_from_bf16(&[0x3f80, 0x4000, 0x4040, 0x4080]),
            0,
            2,
            2,
            2,
            true,
        )
        .expect("creating tensor should succeed");

        let mut dst = Tensor::<F32, _>::new(
            MmapMut::from([0.0, 0.0, 0.0, 0.0].as_slice()),
            0,
            2,
            2,
            2,
            true,
        )
        .expect("creating tensor should succeed");

        dst.cast(&src).expect("`cast` should succeed");

        dst.assert(&[[1.0, 2.0], [3.0, 4.0]]);
    }

    #[test]
    fn case03_rms_norm() {
        let mut x = Tensor::<F32, _>::new(
            MmapMut::from([3.0, 4.0, 0.0, 5.0].as_slice()),
            0,
            2,
            2,
            2,
            true,
        )
        .expect("creating tensor should succeed");

        let w = Tensor::<F32, _>::new(Mmap::from([2.0, 0.5].as_slice()), 0, 1, 2, 2, true)
            .expect("creating tensor should succeed");

        x.rms_norm(&w, 0.0)
            .expect("applying rms_norm should succeed");

        x.assert(&[[1.6970563, 0.56568545], [0.0, 0.70710677]]);
    }
}
