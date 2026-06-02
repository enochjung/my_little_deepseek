use crate::kernel;
use crate::storage::{Mmap, MmapMut, Storage};
use std::marker::PhantomData;
use std::ops::Range;

pub(crate) trait Ownership {}
pub(crate) struct Own;
impl Ownership for Own {}
pub(crate) struct Mut;
impl Ownership for Mut {}
pub(crate) struct Ref;
impl Ownership for Ref {}

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

pub(crate) trait Location {}
pub(crate) struct Host;
impl Location for Host {}
#[allow(unused)]
pub(crate) struct Device;
impl Location for Device {}

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
}

pub(crate) trait StorageType<'a, L: Location> {
    type Memory: Storage;
}
impl<'a> StorageType<'a, Host> for Own {
    type Memory = Mmap;
}
impl<'a> StorageType<'a, Host> for Mut {
    type Memory = MmapMut;
}
impl<'a> StorageType<'a, Host> for Ref {
    type Memory = &'a Mmap;
}
fn ref_mem_from_mmap<'s>(mem: &'s Mmap) -> <Ref as StorageType<'s, Host>>::Memory {
    mem
}

pub(crate) struct Tensor<'a, O, E, L>
where
    O: Ownership + StorageType<'a, L>,
    E: ElemType,
    L: Location,
{
    data: <O as StorageType<'a, L>>::Memory,
    layout: Layout,
    _phantom: PhantomData<E>,
}

impl<'a, O, E, L> Tensor<'a, O, E, L>
where
    O: Ownership + StorageType<'a, L>,
    E: ElemType,
    L: Location,
{
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
}

impl<'a, E> Tensor<'a, Own, E, Host>
where
    E: ElemType,
{
    pub(crate) fn slice<'s>(
        &'s self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Tensor<'s, Ref, E, Host>
    where
        Ref: StorageType<'s, Host>,
    {
        let data = ref_mem_from_mmap(&self.data);
        Tensor {
            data,
            layout: self.sliced_layout(rows, cols),
            _phantom: PhantomData,
        }
    }
}

impl<'a, E> Tensor<'a, Mut, E, Host>
where
    E: ElemType,
{
    pub(crate) fn slice<'s>(
        &'s self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Tensor<'s, Ref, E, Host> {
        let data = ref_mem_from_mmap(self.data.as_const_mmap());
        Tensor {
            data,
            layout: self.sliced_layout(rows, cols),
            _phantom: PhantomData,
        }
    }

    pub(crate) fn to_readonly(self) -> Tensor<'a, Own, E, Host> {
        Tensor {
            data: self.data.into(),
            layout: self.layout,
            _phantom: PhantomData,
        }
    }
}

impl<'a, E> Tensor<'a, Ref, E, Host>
where
    E: ElemType,
{
    pub(crate) fn slice<'s>(
        &'s self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Tensor<'s, Ref, E, Host>
    where
        'a: 's,
    {
        let data = ref_mem_from_mmap(self.data);
        Tensor {
            data,
            layout: self.sliced_layout(rows, cols),
            _phantom: PhantomData,
        }
    }
}

impl<E> Tensor<'_, Own, E, Host>
where
    E: ElemType,
{
    pub(crate) fn new(
        data: Mmap,
        nrow: u32,
        ncol: u32,
        is_row_major: bool,
    ) -> Result<Self, crate::Error> {
        let stride = if is_row_major { ncol } else { nrow };
        let layout = Layout {
            offset: 0,
            nrow,
            ncol,
            stride,
            is_row_major,
        };
        validate_space::<E>(data.len(), &layout)?;
        Ok(Self {
            data,
            layout,
            _phantom: PhantomData,
        })
    }
}

impl<E> Tensor<'_, Mut, E, Host>
where
    E: ElemType,
{
    pub(crate) fn new(
        data: MmapMut,
        nrow: u32,
        ncol: u32,
        is_row_major: bool,
    ) -> Result<Self, crate::Error> {
        let stride = if is_row_major { ncol } else { nrow };
        let layout = Layout {
            offset: 0,
            nrow,
            ncol,
            stride,
            is_row_major,
        };
        validate_space::<E>(data.len(), &layout)?;
        Ok(Self {
            data,
            layout,
            _phantom: PhantomData,
        })
    }
}
impl<'a> Tensor<'a, Mut, F32, Host> {
    pub(crate) fn copy<'b, O>(
        &mut self,
        row_idx: u32,
        col_idx: u32,
        other: &Tensor<'b, O, F32, Host>,
    ) -> Result<(), crate::Error>
    where
        O: Ownership + StorageType<'b, Host>,
    {
        if self.layout.is_row_major != other.layout.is_row_major {
            return Err(crate::Error::operation_not_supported(
                "copy with different layout majority",
            ));
        }
        if row_idx + other.layout.nrow > self.layout.nrow {
            return Err(crate::Error::out_of_bound(
                (row_idx + other.layout.nrow) as usize,
                self.layout.nrow as usize,
            ));
        }
        if col_idx + other.layout.ncol > self.layout.ncol {
            return Err(crate::Error::out_of_bound(
                (col_idx + other.layout.ncol) as usize,
                self.layout.ncol as usize,
            ));
        }

        let n_lines = other.layout.n_lines();
        let line_elems = other.layout.line_elems();
        let dst_offset = self.layout.index(row_idx, col_idx) as usize * F32::BYTES;
        let len = line_elems as usize * F32::BYTES;
        let dst_stride = self.layout.stride as usize * F32::BYTES;
        let src_stride = other.layout.stride as usize * F32::BYTES;
        let mut dst = unsafe { self.data.as_mut_ptr().byte_add(dst_offset) };
        let mut src = unsafe { other.data.as_ptr().byte_add(other.layout.offset) };
        for _ in 0..n_lines {
            unsafe { kernel::copy(dst, src, len) };
            dst = unsafe { dst.byte_add(dst_stride) };
            src = unsafe { src.byte_add(src_stride) };
        }
        Ok(())
    }

    pub(crate) fn cast<'b, O>(
        &mut self,
        other: &Tensor<'b, O, BF16, Host>,
    ) -> Result<(), crate::Error>
    where
        O: Ownership + StorageType<'b, Host>,
    {
        validate_shape(
            other.layout.nrow,
            other.layout.ncol,
            self.layout.nrow,
            self.layout.ncol,
        )?;

        let dst = self.data.as_mut_ptr() as *mut f32;
        let src = unsafe { other.data.as_ptr().byte_add(other.layout.offset) };
        let is_src_packed = other.layout.stride == other.layout.line_elems();
        if is_src_packed {
            let n = self.layout.nrow as usize * self.layout.ncol as usize;
            unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
        } else {
            let mut dst = dst;
            let mut src = src;
            let n_lines = self.layout.n_lines();
            let line_elems = self.layout.line_elems();
            for _ in 0..n_lines {
                unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, line_elems as usize) };
                dst = unsafe { dst.add(line_elems as usize) };
                src = unsafe { src.byte_add(other.layout.stride as usize * BF16::BYTES) };
            }
        }
        Ok(())
    }

    /*

    pub(crate) fn residual<T: TensorTrait<F32, Host> + ?Sized>(
        &mut self,
        _other: &T,
    ) -> Result<(), crate::Error> {
        todo!()
        let is_row_major = self.layout().is_row_major;
        let nrow = self.layout().nrow;
        let ncol = self.layout().ncol;
        let stride = self.layout().stride;
        let n_lines = self.layout().n_lines();
        let line_bytes = self.layout().line_bytes();
        let is_packed = self.layout().is_packed();
        let other_layout = other.layout();

        if is_row_major != other_layout.is_row_major {
            return Err(crate::Error::operation_not_supported(
                "residual with mismatched tensor layout majority (row-major vs col-major)",
            ));
        }

        if nrow != other_layout.nrow {
            return Err(crate::Error::shape_mismatch(
                nrow as usize,
                other_layout.nrow as usize,
            ));
        }
        if ncol != other_layout.ncol {
            return Err(crate::Error::shape_mismatch(
                ncol as usize,
                other_layout.ncol as usize,
            ));
        }

        let n = nrow as usize * ncol as usize;
        if n == 0 {
            return Ok(());
        }

        let dst_stride_bytes = stride as usize * F32::BYTES;
        let src_stride_bytes = other_layout.stride as usize * F32::BYTES;

        unsafe {
            let y = self.as_mut_ptr(0, 0)?;
            let x = other.as_ptr(0, 0);

            if is_packed && other_layout.is_packed() {
                x86_64::add_n_n(y as *mut f32, x as *const f32, n);
            } else {
                let mut y = y;
                let mut x = x;

                for _ in 0..n_lines {
                    x86_64::add_n_n(y as *mut f32, x as *const f32, line_bytes);
                    y = y.add(dst_stride_bytes);
                    x = x.add(src_stride_bytes);
                }
            }
        }

        Ok(())
    }

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

    pub fn rms_norm<'b, O>(
        &mut self,
        weight: &Tensor<'b, O, F32, Host>,
        epsilon: f32,
    ) -> Result<(), crate::Error>
    where
        O: Ownership + StorageType<'b, Host>,
    {
        if self.layout.is_row_major != true {
            return Err(crate::Error::operation_not_supported(
                "rms-norm with column-major",
            ));
        }
        validate_shape(weight.layout.nrow, weight.layout.ncol, 1, self.layout.ncol)?;

        let n_lines = self.layout.n_lines();
        let line_elems = self.layout.line_elems();
        let mut y = self.data.as_mut_ptr() as *mut f32;
        let x = unsafe { weight.data.as_ptr().byte_add(weight.layout.offset) } as *const f32;
        for _ in 0..n_lines {
            let rms = unsafe { kernel::rms_n(y as *const f32, line_elems as usize) };
            let scale = 1.0 / (rms + epsilon);
            unsafe { kernel::mul_n_n(y, x, scale, line_elems as usize) };
            y = unsafe { y.add(self.layout.stride as usize) };
        }
        Ok(())
    }

    /*
    pub(crate) fn muladd_weight_bias<T: TensorTrait<F32, Host> + ?Sized>(
        &mut self,
        _weight: &T,
        _bias: &T,
    ) -> Result<(), crate::Error> {
        todo!()
        let [m, n] = self.shape();
        let w_shape = weight.shape();
        let b_shape = bias.shape();

        if w_shape[0] != n || w_shape[1] != n {
            return Err(crate::Error::shape_mismatch(
                n as usize,
                w_shape[0] as usize,
            ));
        }

        if !(b_shape[0] == 1 && b_shape[1] == n) {
            return Err(crate::Error::shape_mismatch(1, b_shape[0] as usize));
        }

        if !self.layout().is_row_major() || !weight.layout().is_row_major() {
            return Err(crate::Error::operation_not_supported(
                "muladd_weight_bias requires row-major matrices",
            ));
        }

        unsafe {
            let c_base = self.as_mut_ptr(0, 0)? as *mut f32;
            let a_base = weight.as_ptr(0, 0) as *const f32;
            let b_base = bias.as_ptr(0, 0) as *const f32;

            // temporary buffer for one row (or reused across rows)
            let mut buf: Vec<f32> = vec![0.0f32; n as usize];
            let buf_ptr = buf.as_mut_ptr();

            x86_64::muladd_mn_nn_1n(
                c_base,
                true,
                self.layout().stride() as usize,
                a_base,
                true,
                weight.layout().stride() as usize,
                b_base,
                buf_ptr,
                m as usize,
                n as usize,
            );
        }

        Ok(())
    }

     */
}

impl<'a, E> Tensor<'a, Ref, E, Host>
where
    E: ElemType,
{
    pub(crate) fn new(
        data: &'a Mmap,
        offset: usize,
        nrow: u32,
        ncol: u32,
        is_row_major: bool,
    ) -> Result<Self, crate::Error> {
        let stride = if is_row_major { ncol } else { nrow };
        let layout = Layout {
            offset,
            nrow,
            ncol,
            stride,
            is_row_major,
        };
        validate_space::<E>(data.len(), &layout)?;
        Ok(Self {
            data,
            layout,
            _phantom: PhantomData,
        })
    }
}

fn validate_space<E: ElemType>(data_len: usize, layout: &Layout) -> Result<(), crate::Error> {
    if layout.n_lines() == 0 {
        return Ok(());
    }
    let required =
        layout.index(layout.n_lines() - 1, layout.line_elems()) as usize * E::BYTES + layout.offset;
    if data_len < required {
        return Err(crate::Error::insufficient_storage_space(required, data_len));
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
impl Tensor<'_, Ref, F32, Host> {
    pub(crate) fn assert(&self, answer: &[f32]) -> () {
        let size = self.layout.nrow as usize * self.layout.ncol as usize;
        assert_eq!(size, answer.len(), "invalid test data");

        let n_lines = self.layout.n_lines();
        let line_elems = self.layout.line_elems();
        let ptr = self.data.as_ptr() as *const f32;
        let mut idx = 0;

        for i in 0..n_lines as usize {
            for j in 0..line_elems as usize {
                let expected = answer[idx];
                let actual = unsafe { *(ptr.add(i * self.layout.stride as usize + j)) };
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "row {} col {} mismatch: actual {}, expected {}, diff {}",
                    i,
                    j,
                    actual,
                    expected,
                    (actual - expected).abs()
                );
                idx += 1;
            }
        }
    }
}

/*
impl<N: Numeric, SM: StorageMut> TensorMut<N, SM> {
    pub(crate) fn take_storage(self) -> SM {
        self.storage
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    fn mmap_mut_from_f32(data: &[f32]) -> MmapMut {
        let len = data.len() * F32::BYTES;
        let mut mmap = MmapMut::new(len).expect("allocating mmap should succeed");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mmap.as_mut_ptr() as *mut f32, data.len())
        };
        mmap
    }

    fn mmap_mut_from_bf16(data: &[u16]) -> MmapMut {
        let len = data.len() * BF16::BYTES;
        let mut mmap = MmapMut::new(len).expect("allocating mmap should succeed");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mmap.as_mut_ptr() as *mut u16, data.len())
        };
        mmap
    }

    #[test]
    fn case01_copy_subtensor() {
        let src_mem = mmap_mut_from_f32(&[1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        let src = Tensor::<Mut, F32, Host>::new(src_mem, 2, 3, true)
            .expect("creating source tensor should succeed")
            .to_readonly();

        let dst_mem =
            mmap_mut_from_f32(&[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
        let mut dst = Tensor::<Mut, F32, Host>::new(dst_mem, 3, 4, true)
            .expect("creating destination tensor should succeed");

        dst.copy(1, 1, &src)
            .expect("copying subtensor into destination should succeed");

        dst.slice(0..3, 0..4)
            .assert(&[9.0, 9.0, 9.0, 9.0, 9.0, 1.0, 2.0, 3.0, 9.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn case02_cast_bf16_to_f32() {
        let src_mem = mmap_mut_from_bf16(&[0x3f80, 0x4000, 0x4040, 0x4080]);
        let src = Tensor::<Mut, BF16, Host>::new(src_mem, 2, 2, true)
            .expect("creating bf16 source tensor should succeed");

        let dst_mem = mmap_mut_from_f32(&[0.0, 0.0, 0.0, 0.0]);
        let mut dst = Tensor::<Mut, F32, Host>::new(dst_mem, 2, 2, true)
            .expect("creating f32 destination tensor should succeed");

        dst.cast(&src)
            .expect("casting bf16 tensor into f32 tensor should succeed");

        dst.slice(0..2, 0..2).assert(&[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn case03_rms_norm() {
        let x_mem = mmap_mut_from_f32(&[3.0, 4.0, 0.0, 5.0]);
        let mut x = Tensor::<Mut, F32, Host>::new(x_mem, 2, 2, true)
            .expect("creating input tensor for rms_norm should succeed");

        let w_mem = mmap_mut_from_f32(&[2.0, 0.5]);
        let w = Tensor::<Mut, F32, Host>::new(w_mem, 1, 2, true)
            .expect("creating rms_norm weight tensor should succeed")
            .to_readonly();

        x.rms_norm(&w, 0.0)
            .expect("applying rms_norm should succeed");

        x.slice(0..2, 0..2)
            .assert(&[1.6970563, 0.56568545, 0.0, 0.70710677]);
    }
}
