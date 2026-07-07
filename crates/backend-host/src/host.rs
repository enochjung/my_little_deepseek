use common::{Backend, ElemType, MatrixLayout};

use crate::mem::HostMem;

use core::marker::PhantomData;

pub struct Host<T: ElemType> {
    _phantom: PhantomData<T>,
}

#[cfg(target_feature = "avx512f")]
use crate::kernel::avx512_f32 as KernelF32;
#[cfg(not(target_feature = "avx512f"))]
use crate::kernel::unknown_f32 as KernelF32;

fn as_ptr<T: ElemType>(mem: &HostMem<T>, layout: &MatrixLayout) -> *const T {
    let stride = mem.ncol;
    let offset = (layout.srow as usize * stride as usize + layout.scol as usize) * size_of::<T>();
    unsafe { mem.mmap_ptr.byte_add(offset) }
}

fn as_mut_ptr<T: ElemType>(mem: &mut HostMem<T>, layout: &MatrixLayout) -> *mut T {
    as_ptr(mem, layout) as *mut T
}

fn is_packed<T: ElemType>(mem: &HostMem<T>, layout: &MatrixLayout) -> bool {
    mem.ncol == layout.ncol
}

impl Backend for Host<f32> {
    type Item = f32;
    type Mem = HostMem<f32>;

    unsafe fn elem_add_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        if is_packed(dst_mem, dst_layout)
            && is_packed(src_mem, src_layout)
            && dst_layout.is_trans == src_layout.is_trans
        {
            let len = (dst_layout.nrow as usize) * (dst_layout.ncol as usize);
            unsafe { KernelF32::elem_add_assign(dst, src, len) };
        } else {
            let ldy = dst_mem.ncol;
            let lda = src_mem.ncol;
            let m = dst_layout.nrow as usize;
            let n = dst_layout.ncol as usize;

            match (dst_layout.is_trans, src_layout.is_trans) {
                (false, false) => unsafe {
                    KernelF32::elem_add_assign_rmn_rmn(dst, ldy, src, lda, m, n)
                },
                (false, true) => unsafe {
                    KernelF32::elem_add_assign_rmn_cmn(dst, ldy, src, lda, m, n)
                },
                (true, true) => unsafe {
                    KernelF32::elem_add_assign_rmn_rmn(dst, ldy, src, lda, n, m)
                },
                (true, false) => unsafe {
                    KernelF32::elem_add_assign_rmn_cmn(dst, ldy, src, lda, n, m)
                },
            }
        }
    }

    // safety:
    // - src_layout.nrow == 1
    // - dst_layout.ncol == src_layout.ncol
    unsafe fn elem_br_add_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
    ) {
        let mut dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol as usize;

        for _ in 0..m {
            unsafe { KernelF32::elem_add_assign(dst, src, n) };
            dst = unsafe { dst.add(ldy) };
        }
    }

    unsafe fn argmax(src_mem: &Self::Mem, src_layout: &MatrixLayout) -> u32 {
        let src = as_ptr(src_mem, src_layout);
        let n = src_layout.ncol as usize;
        unsafe { KernelF32::argmax(src, n) }
    }

    unsafe fn copy(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src_mem.ncol;

        if is_packed(dst_mem, dst_layout)
            && is_packed(src_mem, src_layout)
            && dst_layout.is_trans == src_layout.is_trans
        {
            unsafe { KernelF32::copy(dst, src, m * n) };
        } else {
            match (dst_layout.is_trans, src_layout.is_trans) {
                (false, false) => unsafe { KernelF32::copy_rmn_rmn(dst, ldy, src, lda, m, n) },
                (false, true) => unsafe { KernelF32::copy_rmn_cmn(dst, ldy, src, lda, m, n) },
                (true, true) => unsafe { KernelF32::copy_rmn_rmn(dst, ldy, src, lda, n, m) },
                (true, false) => unsafe { KernelF32::copy_rmn_cmn(dst, ldy, src, lda, n, m) },
            }
        }
    }

    unsafe fn fill(dst_mem: &mut Self::Mem, dst_layout: &MatrixLayout, value: f32) {
        let dst = as_mut_ptr(dst_mem, dst_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol as usize;

        if is_packed(dst_mem, dst_layout) {
            unsafe { KernelF32::fill(dst, value, m * n) };
        } else {
            let mut dst = dst;
            for _ in 0..m {
                unsafe { KernelF32::fill(dst, value, n) };
                dst = unsafe { dst.add(ldy) };
            }
        }
    }

    unsafe fn scalar_mul_assign(dst_mem: &mut Self::Mem, dst_layout: &MatrixLayout, value: f32) {
        let dst = as_mut_ptr(dst_mem, dst_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol as usize;

        if is_packed(dst_mem, dst_layout) {
            unsafe { KernelF32::scalar_mul_assign(dst, value, m * n) };
        } else {
            let mut dst = dst;
            for _ in 0..m {
                unsafe { KernelF32::scalar_mul_assign(dst, value, n) };
                dst = unsafe { dst.add(ldy) };
            }
        }
    }

    unsafe fn elem_mul(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src0_mem: &Self::Mem,
        src0_layout: &MatrixLayout,
        src1_mem: &Self::Mem,
        src1_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src0 = as_ptr(src0_mem, src0_layout);
        let src1 = as_ptr(src1_mem, src1_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src0_mem.ncol;
        let ldb = src1_mem.ncol;

        if is_packed(dst_mem, dst_layout)
            && is_packed(src0_mem, src0_layout)
            && is_packed(src1_mem, src1_layout)
            && dst_layout.is_trans == src0_layout.is_trans
            && src0_layout.is_trans == src1_layout.is_trans
        {
            unsafe { KernelF32::elem_mul(dst, src0, src1, m * n) };
        } else {
            match (
                dst_layout.is_trans,
                src0_layout.is_trans,
                src1_layout.is_trans,
            ) {
                (false, false, false) => unsafe {
                    KernelF32::elem_mul_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, false, true) => unsafe {
                    KernelF32::elem_mul_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, false) => unsafe {
                    KernelF32::elem_mul_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, true) => unsafe {
                    KernelF32::elem_mul_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (true, true, true) => unsafe {
                    KernelF32::elem_mul_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, true, false) => unsafe {
                    KernelF32::elem_mul_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, true) => unsafe {
                    KernelF32::elem_mul_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, false) => unsafe {
                    KernelF32::elem_mul_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
            }
        }
    }

    unsafe fn elem_mul_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src_mem.ncol;

        if is_packed(dst_mem, dst_layout)
            && is_packed(src_mem, src_layout)
            && dst_layout.is_trans == src_layout.is_trans
        {
            unsafe { KernelF32::elem_mul_assign(dst, src, m * n) };
        } else {
            match (dst_layout.is_trans, src_layout.is_trans) {
                (false, false) => unsafe {
                    KernelF32::elem_mul_assign_rmn_rmn(dst, ldy, src, lda, m, n)
                },
                (false, true) => unsafe {
                    KernelF32::elem_mul_assign_rmn_cmn(dst, ldy, src, lda, m, n)
                },
                (true, true) => unsafe {
                    KernelF32::elem_mul_assign_rmn_rmn(dst, ldy, src, lda, n, m)
                },
                (true, false) => unsafe {
                    KernelF32::elem_mul_assign_rmn_cmn(dst, ldy, src, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_muladd_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src0_mem: &Self::Mem,
        src0_layout: &MatrixLayout,
        src1_mem: &Self::Mem,
        src1_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src0 = as_ptr(src0_mem, src0_layout);
        let src1 = as_ptr(src1_mem, src1_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src0_mem.ncol;
        let ldb = src1_mem.ncol;

        if is_packed(dst_mem, dst_layout)
            && is_packed(src0_mem, src0_layout)
            && is_packed(src1_mem, src1_layout)
            && dst_layout.is_trans == src0_layout.is_trans
            && src0_layout.is_trans == src1_layout.is_trans
        {
            unsafe { KernelF32::elem_muladd_assign(dst, src0, src1, m * n) };
        } else {
            match (
                dst_layout.is_trans,
                src0_layout.is_trans,
                src1_layout.is_trans,
            ) {
                (false, false, false) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, false, true) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, false) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, true) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (true, true, true) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, true, false) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, true) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, false) => unsafe {
                    KernelF32::elem_muladd_assign_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
            }
        }
    }

    unsafe fn elem_mulsub_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src0_mem: &Self::Mem,
        src0_layout: &MatrixLayout,
        src1_mem: &Self::Mem,
        src1_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src0 = as_ptr(src0_mem, src0_layout);
        let src1 = as_ptr(src1_mem, src1_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src0_mem.ncol;
        let ldb = src1_mem.ncol;

        if is_packed(dst_mem, dst_layout)
            && is_packed(src0_mem, src0_layout)
            && is_packed(src1_mem, src1_layout)
            && dst_layout.is_trans == src0_layout.is_trans
            && src0_layout.is_trans == src1_layout.is_trans
        {
            unsafe { KernelF32::elem_mulsub_assign(dst, src0, src1, m * n) };
        } else {
            match (
                dst_layout.is_trans,
                src0_layout.is_trans,
                src1_layout.is_trans,
            ) {
                (false, false, false) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, false, true) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, false) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (false, true, true) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, m, n)
                },
                (true, true, true) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_rmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, true, false) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_rmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, true) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_cmn_rmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
                (true, false, false) => unsafe {
                    KernelF32::elem_mulsub_assign_rmn_cmn_cmn(dst, ldy, src0, lda, src1, ldb, n, m)
                },
            }
        }
    }

    unsafe fn matmul(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src0_mem: &Self::Mem,
        src0_layout: &MatrixLayout,
        src1_mem: &Self::Mem,
        src1_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src0 = as_ptr(src0_mem, src0_layout);
        let src1 = as_ptr(src1_mem, src1_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let k = if src0_layout.is_trans {
            src0_layout.nrow as usize
        } else {
            src0_layout.ncol as usize
        };
        let ldy = dst_mem.ncol;
        let lda = src0_mem.ncol;
        let ldb = src1_mem.ncol;

        if !dst_layout.is_trans {
            match (src0_layout.is_trans, src1_layout.is_trans) {
                (false, false) => unsafe {
                    KernelF32::matmul_rmn_rmk_rkn(dst, ldy, src0, lda, src1, ldb, m, k, n)
                },
                (false, true) => unsafe {
                    KernelF32::matmul_rmn_rmk_ckn(dst, ldy, src0, lda, src1, ldb, m, k, n)
                },
                (true, false) => unsafe {
                    KernelF32::matmul_rmn_cmk_rkn(dst, ldy, src0, lda, src1, ldb, m, k, n)
                },
                (true, true) => unsafe {
                    KernelF32::matmul_rmn_cmk_ckn(dst, ldy, src0, lda, src1, ldb, m, k, n)
                },
            }
        } else {
            match (src0_layout.is_trans, src1_layout.is_trans) {
                (false, false) => unsafe {
                    KernelF32::matmul_rmn_cmk_ckn(dst, ldy, src1, ldb, src0, lda, n, k, m)
                },
                (false, true) => unsafe {
                    KernelF32::matmul_rmn_rmk_ckn(dst, ldy, src1, ldb, src0, lda, n, k, m)
                },
                (true, false) => unsafe {
                    KernelF32::matmul_rmn_cmk_rkn(dst, ldy, src1, ldb, src0, lda, n, k, m)
                },
                (true, true) => unsafe {
                    KernelF32::matmul_rmn_rmk_rkn(dst, ldy, src1, ldb, src0, lda, n, k, m)
                },
            }
        }
    }

    unsafe fn matmul_assign(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol;
        let lda = src_mem.ncol;

        match (dst_layout.is_trans, src_layout.is_trans) {
            (false, false) => unsafe { KernelF32::matmul_assign_rmn_rnn(dst, ldy, src, lda, m, n) },
            (false, true) => unsafe { KernelF32::matmul_assign_rmn_cnn(dst, ldy, src, lda, m, n) },
            (true, false) => unsafe { KernelF32::matmul_assign_cmn_rnn(dst, ldy, src, lda, m, n) },
            (true, true) => unsafe { KernelF32::matmul_assign_cmn_cnn(dst, ldy, src, lda, m, n) },
        }
    }

    unsafe fn rms_norm(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        src_mem: &Self::Mem,
        src_layout: &MatrixLayout,
        epsilon: Self::Item,
    ) {
        let mut dst = as_mut_ptr(dst_mem, dst_layout);
        let src = as_ptr(src_mem, src_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol as usize;

        for _ in 0..m {
            let rms_val = unsafe { KernelF32::rms(dst, n) };
            let multiplier = 1.0 / (rms_val + epsilon);
            unsafe { KernelF32::scalar_mul_assign(dst, multiplier, n) };
            unsafe { KernelF32::elem_mul_assign(dst, src, n) };
            dst = unsafe { dst.add(ldy) };
        }
    }

    unsafe fn rope_cos(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        k: Self::Item,
        theta: Self::Item,
        d: Self::Item,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let n = dst_layout.ncol as usize;
        unsafe { KernelF32::rope_cos(dst, k, theta, d, n) };
    }

    unsafe fn rope_sin(
        dst_mem: &mut Self::Mem,
        dst_layout: &MatrixLayout,
        k: Self::Item,
        theta: Self::Item,
        d: Self::Item,
    ) {
        let dst = as_mut_ptr(dst_mem, dst_layout);
        let n = dst_layout.ncol as usize;
        unsafe { KernelF32::rope_sin(dst, k, theta, d, n) };
    }

    unsafe fn masked_safe_softmax(dst_mem: &mut Self::Mem, dst_layout: &MatrixLayout, n_mask: u32) {
        let dst = as_mut_ptr(dst_mem, dst_layout);

        let n = dst_layout.ncol as usize;
        let n_mask = n_mask as usize;

        unsafe { KernelF32::masked_safe_softmax(dst, n_mask, n) };
    }

    unsafe fn silu(dst_mem: &mut Self::Mem, dst_layout: &MatrixLayout) {
        let dst = as_mut_ptr(dst_mem, dst_layout);

        let m = dst_layout.nrow as usize;
        let n = dst_layout.ncol as usize;
        let ldy = dst_mem.ncol as usize;

        if is_packed(dst_mem, dst_layout) {
            unsafe { KernelF32::silu(dst, m * n) };
        } else {
            let mut dst = dst;
            for _ in 0..m {
                unsafe { KernelF32::silu(dst, n) };
                dst = unsafe { dst.add(ldy) };
            }
        }
    }
}
