//! Fallback scalar kernels for f32

const BLOCK_SIZE: usize = 64;

// y[0..n] += x[0..n]
pub unsafe fn elem_add_assign(y: *mut f32, x: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) += *x.add(i) };
    }
}

// y[0..m, 0..n] += a[0..m, 0..n] (y=row major, a=row major)
pub unsafe fn elem_add_assign_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) += *a.add(i * lda as usize + j) };
        }
    }
}

// y[0..m, 0..n] += a[0..m, 0..n] (y=row major, a=col major)
pub unsafe fn elem_add_assign_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) += *a.add(j * lda as usize + i) };
        }
    }
}

// y[0..n] = a[0..n] + b[0..n]
pub unsafe fn elem_add(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *a.add(i) + *b.add(i) };
    }
}

// y = a + b
pub unsafe fn elem_add_rmn_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) + *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y = a + b
pub unsafe fn elem_add_rmn_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) + *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y = a + b
pub unsafe fn elem_add_rmn_cmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) + *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y = a + b
pub unsafe fn elem_add_rmn_cmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) + *b.add(j * ldb as usize + i)
            };
        }
    }
}

// res = i where max(x[0..n]) == x[i] (any i)
pub unsafe fn argmax(x: *const f32, n: usize) -> u32 {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;
    for i in 0..n {
        let val = unsafe { *x.add(i) };
        if val > max_val {
            max_val = val;
            max_idx = i as u32;
        }
    }
    max_idx
}

// y[0..n] = x[0..n]
pub unsafe fn copy(y: *mut f32, x: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *x.add(i) };
    }
}

// y[0..n] = x[0..n]
pub unsafe fn copy_rmn_rmn(y: *mut f32, ldy: u32, a: *const f32, lda: u32, m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = *a.add(i * lda as usize + j) };
        }
    }
}

// y[0..n] = x[0..n]
pub unsafe fn copy_rmn_cmn(y: *mut f32, ldy: u32, a: *const f32, lda: u32, m: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) = *a.add(j * lda as usize + i) };
        }
    }
}

// y[0..n] = value
pub unsafe fn fill(y: *mut f32, value: f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = value };
    }
}

// y[0..n] *= value
pub unsafe fn scalar_mul_assign(y: *mut f32, value: f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) *= value };
    }
}

// y[0..n] *= x[0..n]
pub unsafe fn elem_mul_assign(y: *mut f32, x: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) * *x.add(i) };
    }
}

// y[0..n] *= x[0..n]
pub unsafe fn elem_mul_assign_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(i * lda as usize + j) };
        }
    }
}

// y[0..n] *= x[0..n]
pub unsafe fn elem_mul_assign_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(j * lda as usize + i) };
        }
    }
}

// y[0..n] = a[0..n] * b[0..n]
pub unsafe fn elem_mul(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *a.add(i) * *b.add(i) };
    }
}

// y[0..n] = a[0..n] * b[0..n]
pub unsafe fn elem_mul_rmn_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] = a[0..n] * b[0..n]
pub unsafe fn elem_mul_rmn_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y[0..n] = a[0..n] * b[0..n]
pub unsafe fn elem_mul_rmn_cmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] = a[0..n] * b[0..n]
pub unsafe fn elem_mul_rmn_cmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y[0..n] += a[0..n] * b[0..n]
pub unsafe fn elem_muladd_assign(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) + *a.add(i) * *b.add(i) };
    }
}

// y[0..n] += a[0..n] * b[0..n]
pub unsafe fn elem_muladd_assign_rmn_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] += a[0..n] * b[0..n]
pub unsafe fn elem_muladd_assign_rmn_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y[0..n] += a[0..n] * b[0..n]
pub unsafe fn elem_muladd_assign_rmn_cmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] += a[0..n] * b[0..n]
pub unsafe fn elem_muladd_assign_rmn_cmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y[0..n] -= a[0..n] * b[0..n]
pub unsafe fn elem_mulsub_assign(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    for i in 0..n {
        unsafe { *y.add(i) = *y.add(i) - *a.add(i) * *b.add(i) };
    }
}

// y[0..n] -= a[0..n] * b[0..n]
pub unsafe fn elem_mulsub_assign_rmn_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] -= a[0..n] * b[0..n]
pub unsafe fn elem_mulsub_assign_rmn_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// y[0..n] -= a[0..n] * b[0..n]
pub unsafe fn elem_mulsub_assign_rmn_cmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
        }
    }
}

// y[0..n] -= a[0..n] * b[0..n]
pub unsafe fn elem_mulsub_assign_rmn_cmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    n: usize,
) {
    for i in 0..m {
        for j in 0..n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
        }
    }
}

// res = rms(x[0..n])
pub unsafe fn rms(x: *const f32, n: usize) -> f32 {
    let mut sq_sum = 0.0f32;
    for i in 0..n {
        let v = unsafe { *x.add(i) };
        sq_sum += v * v;
    }
    (sq_sum / (n as f32)).sqrt()
}

// y = cos( k / theta^(2*i/d) )
pub unsafe fn rope_cos(y: *mut f32, k: f32, theta: f32, d: f32, n: usize) {
    for i in 0..n {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *y.add(i) = angle.cos() };
    }
}

// y = sin( k / theta^(2*i/d) )
pub unsafe fn rope_sin(y: *mut f32, k: f32, theta: f32, d: f32, n: usize) {
    for i in 0..n {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *y.add(i) = angle.sin() };
    }
}

// y = safe_softmax(y)
pub unsafe fn safe_softmax(y: *mut f32, n: usize) {
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..n {
        let v = unsafe { *y.add(i) };
        if v > max_val {
            max_val = v;
        }
    }

    let mut sum = 0.0;
    for i in 0..n {
        let exp_v = (unsafe { *y.add(i) } - max_val).exp();
        unsafe { *y.add(i) = exp_v };
        sum += exp_v;
    }

    let multiplier = 1.0 / sum;
    for i in 0..n {
        unsafe { *y.add(i) *= multiplier };
    }
}

// y = silu(y)
pub unsafe fn silu(y: *mut f32, n: usize) {
    for i in 0..n {
        let v = unsafe { *y.add(i) };
        unsafe { *y.add(i) = v / (1.0 + (-v).exp()) };
    }
}

// y(m*n) = a(m*k) @ b(k*n)
// row/row/row major
pub unsafe fn matmul_rmn_rmk_rkn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
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
}

// y(m*n) = a(m*k) @ b(k*n)
// row/row/col major
pub unsafe fn matmul_rmn_rmk_ckn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
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
}

// y(m*n) = a(m*k) @ b(k*n)
// row/col/row major
pub unsafe fn matmul_rmn_cmk_rkn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
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
}

pub unsafe fn matmul_rmn_cmk_ckn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) {
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
}

// y(m*n) = y(m*n) @ a(n*n)
// row/row major
pub unsafe fn matmul_assign_rmn_rnn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
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
}

pub unsafe fn matmul_assign_rmn_cnn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
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
}

pub unsafe fn matmul_assign_cmn_rnn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
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
}

pub unsafe fn matmul_assign_cmn_cnn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
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
}
