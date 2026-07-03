use common::{Backend, MatrixLayout, Memory, MemoryMut};

use crate::mem::HostMem;

#[cfg(target_feature = "avx512f")]
use crate::kernel::avx512_f32 as Kernel;
#[cfg(not(target_feature = "avx512f"))]
use crate::kernel::unknown_f32 as Kernel;

pub struct HostF32;

impl Backend for HostF32 {
    type Item = f32;
    type Mem = HostMem<f32>;

    unsafe fn elem_add_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src_ml.row_stride
        } else {
            src_ml.col_stride
        };

        if y_rm && a_rm && ldy == n as u32 && lda == n as u32
            || !y_rm && !a_rm && ldy == m as u32 && lda == m as u32
        {
            unsafe { Kernel::elem_add_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_br_add_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        } as usize;

        if y_rm {
            for i in 0..m {
                unsafe { Kernel::elem_add_assign(dst_ptr.add(i * ldy), src_ptr, n) };
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    unsafe { Kernel::elem_add_assign(dst_ptr.add(i + j * ldy), src_ptr.add(j), 1) };
                }
            }
        }
    }

    unsafe fn argmax(src: &Self::Mem, src_ml: &MatrixLayout<f32>) -> u32 {
        let n = src_ml.ncol as usize;
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };
        unsafe { Kernel::argmax(src_ptr, n) }
    }

    unsafe fn copy(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src_ml.row_stride
        } else {
            src_ml.col_stride
        };

        if y_rm && a_rm && ldy == n as u32 && lda == n as u32
            || !y_rm && !a_rm && ldy == m as u32 && lda == m as u32
        {
            unsafe { Kernel::copy(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe { Kernel::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n) },
                (true, false) => unsafe { Kernel::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n) },
                (false, false) => unsafe { Kernel::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m) },
                (false, true) => unsafe { Kernel::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m) },
            }
        }
    }

    unsafe fn fill(dst: &mut Self::Mem, dst_ml: &MatrixLayout<f32>, value: f32) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        } as usize;

        if y_rm && ldy == n || !y_rm && ldy == m {
            unsafe { Kernel::fill(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::fill(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::fill(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn scalar_mul_assign(dst: &mut Self::Mem, dst_ml: &MatrixLayout<f32>, value: f32) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        } as usize;

        if y_rm && ldy == n || !y_rm && ldy == m {
            unsafe { Kernel::scalar_mul_assign(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::scalar_mul_assign(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::scalar_mul_assign(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn elem_mul(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src0: &Self::Mem,
        src0_ml: &MatrixLayout<f32>,
        src1: &Self::Mem,
        src1_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src0_ptr = unsafe { src0.as_ptr().byte_add(src0_ml.offset) };
        let src1_ptr = unsafe { src1.as_ptr().byte_add(src1_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src0_ml.col_stride == 1;
        let b_rm = src1_ml.col_stride == 1;

        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src0_ml.row_stride
        } else {
            src0_ml.col_stride
        };
        let ldb = if b_rm {
            src1_ml.row_stride
        } else {
            src1_ml.col_stride
        };

        if y_rm && a_rm && b_rm && ldy == n as u32 && lda == n as u32 && ldb == n as u32
            || !y_rm && !a_rm && !b_rm && ldy == m as u32 && lda == m as u32 && ldb == m as u32
        {
            unsafe { Kernel::elem_mul(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::elem_mul_rmn_rmn_rmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n)
                },
                (true, true, false) => unsafe {
                    Kernel::elem_mul_rmn_rmn_cmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n)
                },
                (true, false, true) => unsafe {
                    Kernel::elem_mul_rmn_cmn_rmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n)
                },
                (true, false, false) => unsafe {
                    Kernel::elem_mul_rmn_cmn_cmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n)
                },
                (false, false, false) => unsafe {
                    Kernel::elem_mul_rmn_rmn_rmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m)
                },
                (false, false, true) => unsafe {
                    Kernel::elem_mul_rmn_rmn_cmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m)
                },
                (false, true, false) => unsafe {
                    Kernel::elem_mul_rmn_cmn_rmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m)
                },
                (false, true, true) => unsafe {
                    Kernel::elem_mul_rmn_cmn_cmn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m)
                },
            }
        }
    }

    unsafe fn elem_mul_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src_ml.row_stride
        } else {
            src_ml.col_stride
        };

        if y_rm && a_rm && ldy == n as u32 && lda == n as u32
            || !y_rm && !a_rm && ldy == m as u32 && lda == m as u32
        {
            unsafe { Kernel::elem_mul_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_muladd_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src0: &Self::Mem,
        src0_ml: &MatrixLayout<f32>,
        src1: &Self::Mem,
        src1_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src0_ptr = unsafe { src0.as_ptr().byte_add(src0_ml.offset) };
        let src1_ptr = unsafe { src1.as_ptr().byte_add(src1_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src0_ml.col_stride == 1;
        let b_rm = src1_ml.col_stride == 1;

        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src0_ml.row_stride
        } else {
            src0_ml.col_stride
        };
        let ldb = if b_rm {
            src1_ml.row_stride
        } else {
            src1_ml.col_stride
        };

        if y_rm && a_rm && b_rm && ldy == n as u32 && lda == n as u32 && ldb == n as u32
            || !y_rm && !a_rm && !b_rm && ldy == m as u32 && lda == m as u32 && ldb == m as u32
        {
            unsafe { Kernel::elem_muladd_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn elem_mulsub_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src0: &Self::Mem,
        src0_ml: &MatrixLayout<f32>,
        src1: &Self::Mem,
        src1_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src0_ptr = unsafe { src0.as_ptr().byte_add(src0_ml.offset) };
        let src1_ptr = unsafe { src1.as_ptr().byte_add(src1_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src0_ml.col_stride == 1;
        let b_rm = src1_ml.col_stride == 1;

        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src0_ml.row_stride
        } else {
            src0_ml.col_stride
        };
        let ldb = if b_rm {
            src1_ml.row_stride
        } else {
            src1_ml.col_stride
        };

        if y_rm && a_rm && b_rm && ldy == n as u32 && lda == n as u32 && ldb == n as u32
            || !y_rm && !a_rm && !b_rm && ldy == m as u32 && lda == m as u32 && ldb == m as u32
        {
            unsafe { Kernel::elem_mulsub_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn matmul(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src0: &Self::Mem,
        src0_ml: &MatrixLayout<f32>,
        src1: &Self::Mem,
        src1_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let k = src0_ml.ncol as usize;

        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src0_ptr = unsafe { src0.as_ptr().byte_add(src0_ml.offset) };
        let src1_ptr = unsafe { src1.as_ptr().byte_add(src1_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src0_ml.col_stride == 1;
        let b_rm = src1_ml.col_stride == 1;

        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src0_ml.row_stride
        } else {
            src0_ml.col_stride
        };
        let ldb = if b_rm {
            src1_ml.row_stride
        } else {
            src1_ml.col_stride
        };

        if y_rm {
            match (a_rm, b_rm) {
                (true, true) => unsafe {
                    Kernel::matmul_rmn_rmk_rkn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n)
                },
                (true, false) => unsafe {
                    Kernel::matmul_rmn_rmk_ckn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n)
                },
                (false, true) => unsafe {
                    Kernel::matmul_rmn_cmk_rkn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n)
                },
                (false, false) => unsafe {
                    Kernel::matmul_rmn_cmk_ckn(dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n)
                },
            }
        } else {
            match (a_rm, b_rm) {
                (true, true) => unsafe {
                    Kernel::matmul_rmn_cmk_ckn(dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m)
                },
                (true, false) => unsafe {
                    Kernel::matmul_rmn_rmk_ckn(dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m)
                },
                (false, true) => unsafe {
                    Kernel::matmul_rmn_cmk_rkn(dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m)
                },
                (false, false) => unsafe {
                    Kernel::matmul_rmn_rmk_rkn(dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m)
                },
            }
        }
    }

    unsafe fn matmul_assign(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;

        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let a_rm = src_ml.col_stride == 1;

        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        };
        let lda = if a_rm {
            src_ml.row_stride
        } else {
            src_ml.col_stride
        };

        match (y_rm, a_rm) {
            (true, true) => unsafe {
                Kernel::matmul_assign_rmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (true, false) => unsafe {
                Kernel::matmul_assign_rmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, true) => unsafe {
                Kernel::matmul_assign_cmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, false) => unsafe {
                Kernel::matmul_assign_cmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
        }
    }

    unsafe fn rms_norm(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        src: &Self::Mem,
        src_ml: &MatrixLayout<f32>,
        epsilon: f32,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let y = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let a = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let ldy = dst_ml.row_stride as usize;

        for i in 0..m {
            let y = unsafe { y.add(i * ldy) };
            let rms_val = unsafe { Kernel::rms(y, n) };
            let multiplier = 1.0 / (rms_val + epsilon);
            unsafe { Kernel::scalar_mul_assign(y, multiplier, n) };
            unsafe { Kernel::elem_mul_assign(y, a, n) };
        }
    }

    unsafe fn rope_cos(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        k: f32,
        theta: f32,
        d: f32,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::rope_cos(dst_ptr, k, theta, d, n) };
    }

    unsafe fn rope_sin(
        dst: &mut Self::Mem,
        dst_ml: &MatrixLayout<f32>,
        k: f32,
        theta: f32,
        d: f32,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::rope_sin(dst_ptr, k, theta, d, n) };
    }

    unsafe fn masked_safe_softmax(dst: &mut Self::Mem, dst_ml: &MatrixLayout<f32>, n_mask: u32) {
        let n = dst_ml.ncol as usize;
        let n_mask = n_mask as usize;
        let y = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::masked_safe_softmax(y, n_mask, n) };
    }

    unsafe fn silu(dst: &mut Self::Mem, dst_ml: &MatrixLayout<f32>) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };

        let y_rm = dst_ml.col_stride == 1;
        let ldy = if y_rm {
            dst_ml.row_stride
        } else {
            dst_ml.col_stride
        } as usize;

        if y_rm && ldy == n || !y_rm && ldy == m {
            unsafe { Kernel::silu(dst_ptr, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::silu(dst_ptr.add(i * ldy), n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::silu(dst_ptr.add(j * ldy), m) };
                }
            }
        }
    }
}
