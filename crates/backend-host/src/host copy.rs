use core::{Backend, ElemType, MatrixLayout, Memory, MemoryMut};

use crate::kernel::{Kernel, KernelOps};
use crate::mmap::Mmap;

use std::marker::PhantomData;

pub struct Host<T: ElemType> {
    _phantom: PhantomData<T>,
}

impl<T: f32> Backend<f32> for Host<f32> {
    type Memory = Mmap<f32>;

    unsafe fn elem_add_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
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
            unsafe { Kernel::<f32>::elem_add_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<f32>::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<f32>::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<f32>::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<f32>::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_br_add_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
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
                unsafe { Kernel::<f32>::elem_add_assign(dst_ptr.add(i * ldy), src_ptr, n) };
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        Kernel::<f32>::elem_add_assign(dst_ptr.add(i + j * ldy), src_ptr.add(j), 1)
                    };
                }
            }
        }
    }

    unsafe fn argmax<S0: Memory<f32, Base = Self::Memory>>(
        src: &S0,
        src_ml: &MatrixLayout<f32>,
    ) -> u32 {
        let n = src_ml.ncol as usize;
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };
        unsafe { Kernel::<f32>::argmax(src_ptr, n) }
    }

    unsafe fn copy<D: MemoryMut<f32, Base = Self::Memory>, S0: Memory<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
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
            unsafe { Kernel::<f32>::copy(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<f32>::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<f32>::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<f32>::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<f32>::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn fill<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        value: f32,
    ) {
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
            unsafe { Kernel::<f32>::fill(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<f32>::fill(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<f32>::fill(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn scalar_mul_assign<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        value: f32,
    ) {
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
            unsafe { Kernel::<f32>::scalar_mul_assign(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<f32>::scalar_mul_assign(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<f32>::scalar_mul_assign(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn elem_mul<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
        S1: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src0: &S0,
        src0_ml: &MatrixLayout<f32>,
        src1: &S1,
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
            unsafe { Kernel::<f32>::elem_mul(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<f32>::elem_mul_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn elem_mul_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
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
            unsafe { Kernel::<f32>::elem_mul_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<f32>::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<f32>::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<f32>::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<f32>::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_muladd_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
        S1: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src0: &S0,
        src0_ml: &MatrixLayout<f32>,
        src1: &S1,
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
            unsafe { Kernel::<f32>::elem_muladd_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<f32>::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn elem_mulsub_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
        S1: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src0: &S0,
        src0_ml: &MatrixLayout<f32>,
        src1: &S1,
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
            unsafe { Kernel::<f32>::elem_mulsub_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<f32>::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn matmul<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
        S1: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src0: &S0,
        src0_ml: &MatrixLayout<f32>,
        src1: &S1,
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
                    Kernel::<f32>::matmul_rmn_rmk_rkn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (true, false) => unsafe {
                    Kernel::<f32>::matmul_rmn_rmk_ckn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (false, true) => unsafe {
                    Kernel::<f32>::matmul_rmn_cmk_rkn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (false, false) => unsafe {
                    Kernel::<f32>::matmul_rmn_cmk_ckn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
            }
        } else {
            match (a_rm, b_rm) {
                (true, true) => unsafe {
                    Kernel::<f32>::matmul_rmn_cmk_ckn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (true, false) => unsafe {
                    Kernel::<f32>::matmul_rmn_rmk_ckn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (false, true) => unsafe {
                    Kernel::<f32>::matmul_rmn_cmk_rkn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (false, false) => unsafe {
                    Kernel::<f32>::matmul_rmn_rmk_rkn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
            }
        }
    }

    unsafe fn matmul_assign<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
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
                Kernel::<f32>::matmul_assign_rmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (true, false) => unsafe {
                Kernel::<f32>::matmul_assign_rmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, true) => unsafe {
                Kernel::<f32>::matmul_assign_cmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, false) => unsafe {
                Kernel::<f32>::matmul_assign_cmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
        }
    }

    unsafe fn rms_norm<
        D: MemoryMut<f32, Base = Self::Memory>,
        S0: Memory<f32, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        src: &S0,
        src_ml: &MatrixLayout<f32>,
        epsilon: f32,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let dst_ldy = dst_ml.row_stride as usize;
        let src_lda = src_ml.row_stride as usize;

        for i in 0..m {
            let d_ptr = unsafe { dst_ptr.add(i * dst_ldy) };
            let s_ptr = unsafe { src_ptr.add(i * src_lda) };
            let rms_val = unsafe { Kernel::<f32>::rms(d_ptr, n) }.to_f32();
            let multiplier = 1.0 / (rms_val + epsilon);
            unsafe { Kernel::<f32>::scalar_mul_assign(d_ptr, f32::from_f32(multiplier), n) };
            unsafe { Kernel::<f32>::elem_mul_assign(d_ptr, s_ptr, n) };
        }
    }

    unsafe fn rope_cos<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        k: f32,
        theta: f32,
        d: f32,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::<f32>::rope_cos(dst_ptr, k, theta, d, n) };
    }

    unsafe fn rope_sin<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
        k: f32,
        theta: f32,
        d: f32,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::<f32>::rope_sin(dst_ptr, k, theta, d, n) };
    }

    unsafe fn safe_softmax<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let ldy = dst_ml.row_stride as usize;

        for i in 0..m {
            unsafe { Kernel::<f32>::safe_softmax(dst_ptr.add(i * ldy), n) };
        }
    }

    unsafe fn silu<D: MemoryMut<f32, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f32>,
    ) {
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
            unsafe { Kernel::<f32>::silu(dst_ptr, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<f32>::silu(dst_ptr.add(i * ldy), n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<f32>::silu(dst_ptr.add(j * ldy), m) };
                }
            }
        }
    }
}

/*
impl<T: ElemType> Backend<T> for crate::Host<T> {
    type Memory = Mmap<T>;

    unsafe fn elem_add_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::elem_add_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<T>::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<T>::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<T>::elem_add_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<T>::elem_add_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_br_add_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
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
                unsafe { Kernel::<T>::elem_add_assign(dst_ptr.add(i * ldy), src_ptr, n) };
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        Kernel::<T>::elem_add_assign(dst_ptr.add(i + j * ldy), src_ptr.add(j), 1)
                    };
                }
            }
        }
    }

    unsafe fn argmax<S0: Memory<T, Base = Self::Memory>>(
        src: &S0,
        src_ml: &MatrixLayout<T>,
    ) -> u32 {
        let n = src_ml.ncol as usize;
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };
        unsafe { Kernel::<T>::argmax(src_ptr, n) }
    }

    unsafe fn copy<D: MemoryMut<T, Base = Self::Memory>, S0: Memory<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::copy(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<T>::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<T>::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<T>::copy_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<T>::copy_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn fill<D: MemoryMut<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        value: T,
    ) {
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
            unsafe { Kernel::<T>::fill(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<T>::fill(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<T>::fill(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn scalar_mul_assign<D: MemoryMut<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        value: T,
    ) {
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
            unsafe { Kernel::<T>::scalar_mul_assign(dst_ptr, value, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<T>::scalar_mul_assign(dst_ptr.add(i * ldy), value, n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<T>::scalar_mul_assign(dst_ptr.add(j * ldy), value, m) };
                }
            }
        }
    }

    unsafe fn elem_mul<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::elem_mul(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<T>::elem_mul_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<T>::elem_mul_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<T>::elem_mul_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<T>::elem_mul_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<T>::elem_mul_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<T>::elem_mul_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<T>::elem_mul_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<T>::elem_mul_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn elem_mul_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::elem_mul_assign(dst_ptr, src_ptr, m * n) };
        } else {
            match (y_rm, a_rm) {
                (true, true) => unsafe {
                    Kernel::<T>::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (true, false) => unsafe {
                    Kernel::<T>::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, m, n)
                },
                (false, false) => unsafe {
                    Kernel::<T>::elem_mul_assign_rmn_rmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
                (false, true) => unsafe {
                    Kernel::<T>::elem_mul_assign_rmn_cmn(dst_ptr, ldy, src_ptr, lda, n, m)
                },
            }
        }
    }

    unsafe fn elem_muladd_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::elem_muladd_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<T>::elem_muladd_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn elem_mulsub_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
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
            unsafe { Kernel::<T>::elem_mulsub_assign(dst_ptr, src0_ptr, src1_ptr, m * n) };
        } else {
            match (y_rm, a_rm, b_rm) {
                (true, true, true) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, true, false) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, true) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (true, false, false) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, n,
                    )
                },
                (false, false, false) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_rmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, false, true) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_rmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, false) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_cmn_rmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
                (false, true, true) => unsafe {
                    Kernel::<T>::elem_mulsub_assign_rmn_cmn_cmn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, n, m,
                    )
                },
            }
        }
    }

    unsafe fn matmul<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
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
                    Kernel::<T>::matmul_rmn_rmk_rkn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (true, false) => unsafe {
                    Kernel::<T>::matmul_rmn_rmk_ckn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (false, true) => unsafe {
                    Kernel::<T>::matmul_rmn_cmk_rkn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
                (false, false) => unsafe {
                    Kernel::<T>::matmul_rmn_cmk_ckn(
                        dst_ptr, ldy, src0_ptr, lda, src1_ptr, ldb, m, k, n,
                    )
                },
            }
        } else {
            match (a_rm, b_rm) {
                (true, true) => unsafe {
                    Kernel::<T>::matmul_rmn_cmk_ckn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (true, false) => unsafe {
                    Kernel::<T>::matmul_rmn_rmk_ckn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (false, true) => unsafe {
                    Kernel::<T>::matmul_rmn_cmk_rkn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
                (false, false) => unsafe {
                    Kernel::<T>::matmul_rmn_rmk_rkn(
                        dst_ptr, ldy, src1_ptr, ldb, src0_ptr, lda, n, k, m,
                    )
                },
            }
        }
    }

    unsafe fn matmul_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
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
                Kernel::<T>::matmul_assign_rmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (true, false) => unsafe {
                Kernel::<T>::matmul_assign_rmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, true) => unsafe {
                Kernel::<T>::matmul_assign_cmn_rnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
            (false, false) => unsafe {
                Kernel::<T>::matmul_assign_cmn_cnn(dst_ptr, ldy, src_ptr, lda, m, n)
            },
        }
    }

    unsafe fn rms_norm<D: MemoryMut<T, Base = Self::Memory>, S0: Memory<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
        epsilon: f32,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src_ptr = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        let dst_ldy = dst_ml.row_stride as usize;
        let src_lda = src_ml.row_stride as usize;

        for i in 0..m {
            let d_ptr = unsafe { dst_ptr.add(i * dst_ldy) };
            let s_ptr = unsafe { src_ptr.add(i * src_lda) };
            let rms_val = unsafe { Kernel::<T>::rms(d_ptr, n) }.to_f32();
            let multiplier = 1.0 / (rms_val + epsilon);
            unsafe { Kernel::<T>::scalar_mul_assign(d_ptr, T::from_f32(multiplier), n) };
            unsafe { Kernel::<T>::elem_mul_assign(d_ptr, s_ptr, n) };
        }
    }

    unsafe fn rope_cos<D: MemoryMut<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        k: T,
        theta: T,
        d: T,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::<T>::rope_cos(dst_ptr, k, theta, d, n) };
    }

    unsafe fn rope_sin<D: MemoryMut<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        k: T,
        theta: T,
        d: T,
    ) {
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        unsafe { Kernel::<T>::rope_sin(dst_ptr, k, theta, d, n) };
    }

    unsafe fn safe_softmax<D: MemoryMut<T, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
    ) {
        let m = dst_ml.nrow as usize;
        let n = dst_ml.ncol as usize;
        let dst_ptr = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let ldy = dst_ml.row_stride as usize;

        for i in 0..m {
            unsafe { Kernel::<T>::safe_softmax(dst_ptr.add(i * ldy), n) };
        }
    }

    unsafe fn silu<D: MemoryMut<T, Base = Self::Memory>>(dst: &mut D, dst_ml: &MatrixLayout<T>) {
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
            unsafe { Kernel::<T>::silu(dst_ptr, m * n) };
        } else {
            if y_rm {
                for i in 0..m {
                    unsafe { Kernel::<T>::silu(dst_ptr.add(i * ldy), n) };
                }
            } else {
                for j in 0..n {
                    unsafe { Kernel::<T>::silu(dst_ptr.add(j * ldy), m) };
                }
            }
        }
    }
}
*/
