//! AVX-512 kernels for bf16

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bf16",
    target_feature = "avx512f"
))]
use std::arch::x86_64::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bf16",
    target_feature = "avx512f"
))]
const BLOCK_SIZE: usize = 64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512bf16",
    target_feature = "avx512f"
))]
pub unsafe fn elem_add_assign(y: *mut bf16, x: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) += *x.add(i) };
    }
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn elem_add_assign_rmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) += *a.add(i * lda as usize + j) };
        }
    }
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn elem_add_assign_rmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) += *a.add(j * lda as usize + i) };
        }
    }
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn argmax(x: *const bf16, n: usize) -> u32 {
    todo!()
    /*
    let mut max_val = bf16::NEG_INFINITY;
    let mut max_idx = 0;
    for i in 0..n {
        let val = unsafe { *x.add(i) };
        if val > max_val {
            max_val = val;
            max_idx = i as u32;
        }
    }
    max_idx
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn copy(y: *mut bf16, x: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = *x.add(i) };
    }
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn copy_rmn_rmn(y: *mut bf16, ldy: u32, a: *const bf16, lda: u32, m: usize, n: usize) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = *a.add(i * lda as usize + j) };
        }
    }
    */
}

#[cfg(target_feature = "avx512bf16")]
pub unsafe fn copy_rmn_cmn(y: *mut bf16, ldy: u32, a: *const bf16, lda: u32, m: usize, n: usize) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = *a.add(j * lda as usize + i) };
        }
    }
    */
}

pub unsafe fn fill(y: *mut bf16, value: bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = value };
    }
    */
}

pub unsafe fn scalar_mul_assign(y: *mut bf16, value: bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) *= value };
    }
    */
}

pub unsafe fn elem_mul_assign(y: *mut bf16, x: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) * *x.add(i) };
    }
    */
}

pub unsafe fn elem_mul_assign_rmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(i * lda as usize + j) };
        }
    }
    */
}

pub unsafe fn elem_mul_assign_rmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(j * lda as usize + i) };
        }
    }
    */
}

pub unsafe fn elem_mul(y: *mut bf16, a: *const bf16, b: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = *a.add(i) * *b.add(i) };
    }
    */
}

pub unsafe fn elem_mul_rmn_rmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_mul_rmn_rmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn elem_mul_rmn_cmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_mul_rmn_cmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn elem_muladd_assign(y: *mut bf16, a: *const bf16, b: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) + *a.add(i) * *b.add(i) };
    }
    */
}

pub unsafe fn elem_muladd_assign_rmn_rmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_muladd_assign_rmn_rmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn elem_muladd_assign_rmn_cmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_muladd_assign_rmn_cmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn elem_mulsub_assign(y: *mut bf16, a: *const bf16, b: *const bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) - *a.add(i) * *b.add(i) };
    }
    */
}

pub unsafe fn elem_mulsub_assign_rmn_rmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_mulsub_assign_rmn_rmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn elem_mulsub_assign_rmn_cmn_rmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
    */
}

pub unsafe fn elem_mulsub_assign_rmn_cmn_cmn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
    */
}

pub unsafe fn rms(x: *const bf16, n: usize) -> bf16 {
    todo!()
    /*
    let mut sq_sum = 0.0bf16;
    for i in 0..n {
        let v = unsafe { *x.add(i) };
        sq_sum += v * v;
    }
    (sq_sum / (n as bf16)).sqrt()
    */
}

pub unsafe fn rope_cos(y: *mut bf16, k: bf16, theta: bf16, d: bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        let angle = k / theta.powf(2.0 * (i as bf16) / d);
        unsafe { *y.add(i) = angle.cos() };
    }
    */
}

pub unsafe fn rope_sin(y: *mut bf16, k: bf16, theta: bf16, d: bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        let angle = k / theta.powf(2.0 * (i as bf16) / d);
        unsafe { *y.add(i) = angle.sin() };
    }
    */
}

pub unsafe fn masked_safe_softmax(y: *mut bf16, n_mask: usize, n: usize) {
    todo!()
    /*
    let end = n - n_mask;

    let mut max_val = bf16::NEG_INFINITY;
    for i in 0..end {
        let v = unsafe { *y.add(i) };
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0;
    for i in 0..end {
        let exp_v = (unsafe { *y.add(i) } - max_val).exp();
        unsafe { *y.add(i) = exp_v };
        sum += exp_v;
    }

    let multiplier = 1.0 / sum;
    for i in 0..end {
        unsafe { *y.add(i) *= multiplier };
    }
    for i in end..n {
        unsafe { *y.add(i) = 0.0 };
    }
    */
}

pub unsafe fn silu(y: *mut bf16, n: usize) {
    todo!()
    /*
    for i in 0..n {
        let v = unsafe { *y.add(i) };
        unsafe { *y.add(i) = v / (1.0 + (-v).exp()) };
    }
    */
}

pub unsafe fn matmul_rmn_rmk_rkn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = 0.0 };
        }
    }
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for pp in (0..k).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let p_end = (pp + BLOCK_SIZE).min(k);
                let j_end = (jj + BLOCK_SIZE).min(n);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;
                        for p in pp..p_end {
                            unsafe {
                                sum += *a.add(i * lda as usize + p) * *b.add(p * ldb as usize + j)
                            };
                        }
                        unsafe { *y.add(i * ldy as usize + j) += sum };
                    }
                }
            }
        }
    }
    */
}

pub unsafe fn matmul_rmn_rmk_ckn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = 0.0 };
        }
    }
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for pp in (0..k).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let p_end = (pp + BLOCK_SIZE).min(k);
                let j_end = (jj + BLOCK_SIZE).min(n);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;
                        for p in pp..p_end {
                            unsafe {
                                sum += *a.add(i * lda as usize + p) * *b.add(j * ldb as usize + p)
                            };
                        }
                        unsafe { *y.add(i * ldy as usize + j) += sum };
                    }
                }
            }
        }
    }
    */
}

pub unsafe fn matmul_rmn_cmk_rkn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = 0.0 };
        }
    }
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for pp in (0..k).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let p_end = (pp + BLOCK_SIZE).min(k);
                let j_end = (jj + BLOCK_SIZE).min(n);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;
                        for p in pp..p_end {
                            unsafe {
                                sum += *a.add(p * lda as usize + i) * *b.add(p * ldb as usize + j)
                            };
                        }
                        unsafe { *y.add(i * ldy as usize + j) += sum };
                    }
                }
            }
        }
    }
    */
}

pub unsafe fn matmul_rmn_cmk_ckn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    b: *const bf16,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
    todo!()
    /*
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = 0.0 };
        }
    }
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for pp in (0..k).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let p_end = (pp + BLOCK_SIZE).min(k);
                let j_end = (jj + BLOCK_SIZE).min(n);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;
                        for p in pp..p_end {
                            unsafe {
                                sum += *a.add(p * lda as usize + i) * *b.add(j * ldb as usize + p)
                            };
                        }
                        unsafe { *y.add(i * ldy as usize + j) += sum };
                    }
                }
            }
        }
    }
    */
}

pub unsafe fn matmul_assign_rmn_rnn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    let mut temp = std::vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            temp[j] = unsafe { *y.add(i * ldy as usize + j) };
        }
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += temp[k] * unsafe { *a.add(k * lda as usize + j) };
            }
            unsafe { *y.add(i * ldy as usize + j) = sum };
        }
    }
    */
}

pub unsafe fn matmul_assign_rmn_cnn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    let mut temp = std::vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            temp[j] = unsafe { *y.add(i * ldy as usize + j) };
        }
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += temp[k] * unsafe { *a.add(j * lda as usize + k) };
            }
            unsafe { *y.add(i * ldy as usize + j) = sum };
        }
    }
    */
}

pub unsafe fn matmul_assign_cmn_rnn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    let mut temp = std::vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            temp[j] = unsafe { *y.add(j * ldy as usize + i) };
        }
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += temp[k] * unsafe { *a.add(k * lda as usize + j) };
            }
            unsafe { *y.add(j * ldy as usize + i) = sum };
        }
    }
    */
}

pub unsafe fn matmul_assign_cmn_cnn(
    y: *mut bf16,
    ldy: u32,
    a: *const bf16,
    lda: u32,
    m: usize,
    n: usize,
) {
    todo!()
    /*
    let mut temp = std::vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            temp[j] = unsafe { *y.add(j * ldy as usize + i) };
        }
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += temp[k] * unsafe { *a.add(j * lda as usize + k) };
            }
            unsafe { *y.add(j * ldy as usize + i) = sum };
        }
    }
    */
}
