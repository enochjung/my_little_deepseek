use super::{Device, DeviceOps, MutableDevice, OwnedDevice};
use crate::kernel;
use crate::tensor::{BF16, ElemType, F32, Layout};
use std::os::fd::AsRawFd;

const PAGE_SIZE: usize = 4096;

#[repr(C)]
pub struct Cpu {
    ptr: *mut (),
    len: usize,
}

unsafe impl Send for Cpu {}
unsafe impl Sync for Cpu {}

impl Cpu {
    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

impl Device for Cpu {
    type Base = Self;

    fn as_base(&self) -> &Self::Base {
        self
    }
    fn len(&self) -> usize {
        self.len
    }
    fn as_ptr(&self) -> *const () {
        self.ptr
    }
}
impl MutableDevice for Cpu {
    fn as_mut_base(&mut self) -> &mut Self::Base {
        self
    }
    fn as_mut_ptr(&mut self) -> *mut () {
        self.ptr
    }
}
impl OwnedDevice for Cpu {
    fn new(len: usize) -> Result<Self, crate::Error> {
        let len = len.max(PAGE_SIZE).next_power_of_two();
        let ptr = new_mmap(len).map_err(crate::Error::io)?;
        Ok(Self { ptr, len })
    }

    fn resize(&mut self, len: usize) -> Result<(), crate::Error> {
        let len = len.max(PAGE_SIZE).next_power_of_two();
        let ptr = resize_mmap(self.ptr, self.len, len).map_err(crate::Error::io)?;
        self.ptr = ptr;
        self.len = len;
        Ok(())
    }
}

impl TryFrom<std::fs::File> for Cpu {
    type Error = crate::Error;

    fn try_from(file: std::fs::File) -> Result<Self, Self::Error> {
        let metadata = file.metadata().map_err(crate::Error::io)?;
        let fd = file.as_raw_fd();
        let len = (metadata.len() as usize).max(PAGE_SIZE);
        let ptr = file_mmap(fd, len).map_err(crate::Error::io)?;
        Ok(Self { ptr, len })
    }
}

impl Drop for Cpu {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

impl DeviceOps<F32> for Cpu {
    unsafe fn add<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_layout.offset) } as *const f32;
        if dst_layout.is_packed() && src_layout.is_packed() {
            let n = (dst_layout.nrow * dst_layout.ncol) as usize;
            unsafe { kernel::add_n_n(dst, src, n) };
        } else {
            let n = dst_layout.ncol as usize;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_layout.nrow {
                unsafe { kernel::add_n_n(dst, src, n) };
                dst = unsafe { dst.add(dst_layout.stride as usize) };
                src = unsafe { src.add(src_layout.stride as usize) };
            }
        }
    }
    unsafe fn argmax<D: Device<Base = Self>>(src: &D, src_layout: &Layout) -> u32 {
        let src = unsafe { src.as_ptr().byte_add(src_layout.offset) } as *mut f32;
        let n = src_layout.ncol as usize;
        unsafe { kernel::argmax_n(src, n) }
    }
    unsafe fn cast_from_bf16<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_layout.offset) };

        if dst_layout.is_packed() && src_layout.is_packed() {
            let n = (dst_layout.nrow * dst_layout.ncol) as usize;
            unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
        } else {
            let n = dst_layout.ncol as usize;
            let src_stride_bytes = src_layout.stride as usize * BF16::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_layout.nrow {
                unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
                dst = unsafe { dst.add(dst_layout.stride as usize) };
                src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }
    }
    unsafe fn copy<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
    ) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) };
        let src = unsafe { src.as_ptr().byte_add(src_layout.offset) };

        if dst_layout.is_packed() && src_layout.is_packed() {
            let len = (dst_layout.nrow * dst_layout.ncol) as usize * F32::BYTES;
            unsafe { kernel::copy(dst, src, len) };
        } else {
            let len = dst_layout.ncol as usize * F32::BYTES;
            let dst_stride_bytes = dst_layout.stride as usize * F32::BYTES;
            let src_stride_bytes = src_layout.stride as usize * F32::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_layout.nrow {
                unsafe { kernel::copy(dst, src, len) };
                dst = unsafe { dst.byte_add(dst_stride_bytes) };
                src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }
    }
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
    ) -> () {
        unsafe { Self::copy(dst, dst_layout, src0, src0_layout) };

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let src1 = unsafe { src1.as_ptr().byte_add(src1_layout.offset) } as *const f32;

        if dst_layout.is_packed() && src1_layout.is_packed() {
            let len = (dst_layout.nrow * dst_layout.ncol) as usize;
            unsafe { kernel::mul_n_n(dst, src1, alpha, len) };
        } else {
            let len = dst_layout.ncol as usize;

            let mut dst = dst;
            let mut src1 = src1;
            for _ in 0..dst_layout.nrow {
                unsafe { kernel::mul_n_n(dst, src1, alpha, len) };
                dst = unsafe { dst.add(dst_layout.stride as usize) };
                src1 = unsafe { src1.add(src1_layout.stride as usize) };
            }
        }
    }
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
    ) -> () {
        let m = src0_layout.nrow as usize;
        let k = src0_layout.ncol as usize;
        let n = src1_layout.ncol as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_layout.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_layout.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_rkn(
                dst,
                dst_layout.stride,
                a,
                src0_layout.stride,
                b,
                src1_layout.stride,
                m,
                k,
                n,
            )
        };
    }
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
    ) -> () {
        let m = src0_layout.nrow as usize;
        let k = src0_layout.ncol as usize;
        let n = src1_layout_t.nrow as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_layout.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_layout_t.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_ckn(
                dst,
                dst_layout.stride,
                a,
                src0_layout.stride,
                b,
                src1_layout_t.stride,
                m,
                k,
                n,
            )
        };
    }
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
    ) -> () {
        let m = src0_layout.nrow as usize;
        let k = src0_layout.ncol as usize;
        let n = src1_layout.ncol as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_layout.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_layout.offset) } as *const f32;
        let c = unsafe { src2.as_ptr().byte_add(src2_layout.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_rkn_r1n(
                dst,
                dst_layout.stride,
                a,
                src0_layout.stride,
                b,
                src1_layout.stride,
                c,
                m,
                k,
                n,
            )
        };
    }
    unsafe fn mul_mn_mk_knt_1n<
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
    ) -> () {
        let m = src0_layout.nrow as usize;
        let k = src0_layout.ncol as usize;
        let n = src1_layout.nrow as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_layout.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_layout.offset) } as *const f32;
        let c = unsafe { src2.as_ptr().byte_add(src2_layout.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_ckn_r1n(
                dst,
                dst_layout.stride,
                a,
                src0_layout.stride,
                b,
                src1_layout.stride,
                c,
                m,
                k,
                n,
            )
        };
    }
    unsafe fn rms_norm<M: MutableDevice<Base = Self>, D0: Device<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        src: &D0,
        src_layout: &Layout,
        epsilon: f32,
    ) -> () {
        let n = dst_layout.ncol as usize;

        let mut dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_layout.offset) } as *const f32;
        for _ in 0..dst_layout.nrow {
            let rms = unsafe { kernel::rms_n(dst as *const f32, n) };
            let scale = 1.0 / (rms + epsilon);
            unsafe { kernel::mul_n_n(dst, src, scale, n) };
            dst = unsafe { dst.add(dst_layout.stride as usize) };
        }
    }
    unsafe fn rope_cos<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        k: f32,
        theta: f32,
        d: f32,
    ) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let n = dst_layout.ncol as usize;
        unsafe { kernel::rope_cos_n(dst, n, k, theta, d) };
    }
    unsafe fn rope_sin<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        k: f32,
        theta: f32,
        d: f32,
    ) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let n = dst_layout.ncol as usize;
        unsafe { kernel::rope_sin_n(dst, n, k, theta, d) };
    }
    unsafe fn silu<M: MutableDevice<Base = Self>>(dst: &mut M, dst_layout: &Layout) -> () {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        let n = (dst_layout.nrow * dst_layout.ncol) as usize;
        unsafe { kernel::silu_n(dst, n) };
    }
    unsafe fn safe_softmax<M: MutableDevice<Base = Self>>(
        dst: &mut M,
        dst_layout: &Layout,
        alpha: f32,
    ) -> () {
        let n = dst_layout.ncol as usize;

        let mut dst = unsafe { dst.as_mut_ptr().byte_add(dst_layout.offset) } as *mut f32;
        for _ in 0..dst_layout.nrow {
            unsafe { kernel::safe_softmax_n(dst, alpha, n) };
            dst = unsafe { dst.add(dst_layout.stride as usize) };
        }
    }
}

fn new_mmap(len: usize) -> Result<*mut (), std::io::Error> {
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "len must be greater than zero",
        ));
    }

    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, -1, 0) };

    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }

    Ok(ptr as *mut ())
}

fn file_mmap(fd: i32, len: usize) -> Result<*mut (), std::io::Error> {
    if len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "len must be greater than zero",
        ));
    }

    let prot = libc::PROT_READ | libc::PROT_WRITE;
    let flags = libc::MAP_PRIVATE;
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, fd, 0) };

    if ptr == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error());
    }

    Ok(ptr as *mut ())
}

fn resize_mmap(ptr: *mut (), prev_len: usize, new_len: usize) -> Result<*mut (), std::io::Error> {
    #[cfg(target_os = "linux")]
    {
        if new_len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "len must be greater than zero",
            ));
        }
        let flags = libc::MREMAP_MAYMOVE;
        let ptr = unsafe { libc::mremap(ptr as *mut libc::c_void, prev_len, new_len, flags) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(ptr as *mut ())
    }

    #[cfg(not(target_os = "linux"))]
    {
        if new_len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "len must be greater than zero",
            ));
        }

        let new_ptr_const = new_mmap(new_len)?;
        let new_ptr = new_ptr_const as *mut u8;

        unsafe {
            let copy_len = std::cmp::min(prev_len, new_len);
            std::ptr::copy_nonoverlapping(ptr as *const u8, new_ptr as *mut u8, copy_len);
            libc::munmap(ptr as *mut libc::c_void, prev_len);
        }

        Ok(new_ptr as *mut ())
    }
}

#[cfg(test)]
impl From<&[f32]> for Cpu {
    /// Test Helper Function
    fn from(value: &[f32]) -> Self {
        let len = value.len() * size_of::<f32>();
        let device = Cpu::new(len).expect("Cpu::new should succeed");

        let dst = device.ptr as *mut f32;
        let src = value.as_ptr();
        let count = value.len();
        unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };

        device
    }
}
