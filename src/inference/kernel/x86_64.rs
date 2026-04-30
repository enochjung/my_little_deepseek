//! x86_64 kernels for vector and matrix primitives.
//!
//! Layout:
//! - Row-major: `base[row * ld + col]`
//! - Column-major: `base[row + col * ld]`

use std::arch::x86_64::*;

/// Computes RMS of `x`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `n > 0`.
pub unsafe fn rms(x: *const f32, n: usize) -> f32 {
    if is_x86_feature_detected!("avx512f") {
        return unsafe { rms_avx512(x, n) };
    }

    unsafe { rms_scalar(x, n) }
}

#[inline]
unsafe fn rms_scalar(x: *const f32, n: usize) -> f32 {
    let mut sq_sum = 0.0f32;
    let mut ptr = x;
    let end = unsafe { x.add(n) };

    while ptr != end {
        let v = unsafe { *ptr };
        sq_sum += v * v;
        ptr = unsafe { ptr.add(1) };
    }
    (sq_sum / (n as f32)).sqrt()
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn rms_avx512(x: *const f32, n: usize) -> f32 {
    let mut ptr = x;
    let mut acc = _mm512_setzero_ps();

    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };

    while ptr != vec_end {
        let xv = unsafe { _mm512_loadu_ps(ptr) };
        let sq = _mm512_mul_ps(xv, xv);
        acc = _mm512_add_ps(acc, sq);
        ptr = unsafe { ptr.add(16) };
    }

    let mut sq_sum = _mm512_reduce_add_ps(acc);
    while ptr != end {
        let v = unsafe { *ptr };
        sq_sum += v * v;
        ptr = unsafe { ptr.add(1) };
    }

    (sq_sum / (n as f32)).sqrt()
}

/// Scales `y` by `alpha * x` element-wise.
///
/// # Safety
///
/// - `x` and `y` must be valid for `n` contiguous `f32` values.
/// - `x` and `y` must not overlap in memory (restrict-like requirement).
/// - `x` and `y` must be aligned to `align_of::<f32>()` (4 bytes).
pub unsafe fn mul(y: *mut f32, x: *const f32, alpha: f32, n: usize) -> () {
    if is_x86_feature_detected!("avx512f") {
        unsafe { mul_avx512(y, x, alpha, n) };
        return;
    }

    unsafe { mul_scalar(y, x, alpha, n) };
}

/// Applies SiLU in place: `x = x / (1 + exp(-x))`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
pub unsafe fn silu(x: *mut f32, n: usize) -> () {
    let mut ptr = x;
    let end = unsafe { x.add(n) };

    while ptr != end {
        let v = unsafe { *ptr };
        unsafe { *ptr = v / (1.0 + (-v).exp()) };
        ptr = unsafe { ptr.add(1) };
    }
}

#[inline]
unsafe fn mul_scalar(y: *mut f32, x: *const f32, alpha: f32, n: usize) -> () {
    let mut y_ptr = y;
    let mut x_ptr = x;
    let x_end = unsafe { x.add(n) };

    while x_ptr != x_end {
        let yv = unsafe { *y_ptr };
        let xv = unsafe { *x_ptr };
        unsafe { *y_ptr = yv * xv * alpha };
        y_ptr = unsafe { y_ptr.add(1) };
        x_ptr = unsafe { x_ptr.add(1) };
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mul_avx512(y: *mut f32, x: *const f32, alpha: f32, n: usize) -> () {
    let mut y_ptr = y;
    let mut x_ptr = x;
    let alpha_v = _mm512_set1_ps(alpha);

    let x_vec_end = unsafe { x.add(n & !15) };
    let x_end = unsafe { x.add(n) };

    while x_ptr != x_vec_end {
        let yv = unsafe { _mm512_loadu_ps(y_ptr as *const f32) };
        let xv = unsafe { _mm512_loadu_ps(x_ptr) };
        let out = _mm512_mul_ps(yv, _mm512_mul_ps(xv, alpha_v));
        unsafe { _mm512_storeu_ps(y_ptr, out) };

        y_ptr = unsafe { y_ptr.add(16) };
        x_ptr = unsafe { x_ptr.add(16) };
    }

    while x_ptr != x_end {
        let yv = unsafe { *y_ptr };
        let xv = unsafe { *x_ptr };
        unsafe { *y_ptr = yv * xv * alpha };
        y_ptr = unsafe { y_ptr.add(1) };
        x_ptr = unsafe { x_ptr.add(1) };
    }
}

/// Computes `C = C * A + B` for one row.
///
/// # Parameters
///
/// - `c`: `C` base pointer, shape `(1, n)`.
/// - `rmc`: `true` if `C` is row-major.
/// - `ldc`: `C` leading dimension.
/// - `a`: `A` base pointer, shape `(n, n)`.
/// - `rma`: `true` if `A` is row-major.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(1, n)`.
/// - `n`: Column count of `C` and `B`, and both dimensions of `A`.
///
/// # Safety
///
/// - `c`, `a`, and `b` cover the required ranges for `n`, layout flags, and strides.
/// - `buf` covers at least `n` `f32` values and is 64-byte aligned.
/// - `c`, `a`, and `b` do not overlap.
/// - `c`, `a`, and `b` are 4-byte aligned.
pub unsafe fn muladd_1n_nn_1n(
    c: *mut f32,
    rmc: bool,
    ldc: usize,
    a: *const f32,
    rma: bool,
    lda: usize,
    b: *const f32,
    buf: *mut f32,
    n: usize,
) -> () {
    let _ = ldc;

    match (rmc, rma) {
        (true, true) => {
            if is_x86_feature_detected!("avx512f") {
                unsafe { muladd_r1n_rnn_1n_avx512(c, a, lda, b, buf, n) }
            } else {
                unsafe { muladd_r1n_rnn_1n_scalar(c, a, lda, b, buf, n) }
            }
        }
        _ => unimplemented!(),
    }
}

#[inline]
unsafe fn muladd_r1n_rnn_1n_scalar(
    c: *mut f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    buf: *mut f32,
    n: usize,
) -> () {
    // Copy `c` into `buf`.
    let mut c_src = c as *const f32;
    let mut c_dst = buf;
    let c_src_end = unsafe { c_src.add(n) };

    while c_src != c_src_end {
        unsafe { *c_dst = *c_src };
        c_src = unsafe { c_src.add(1) };
        c_dst = unsafe { c_dst.add(1) };
    }

    // Load `b` into `c`.
    let mut c_ptr = c;
    let mut b_ptr = b;
    let c_end = unsafe { c.add(n) };

    while c_ptr != c_end {
        unsafe { *c_ptr = *b_ptr };
        c_ptr = unsafe { c_ptr.add(1) };
        b_ptr = unsafe { b_ptr.add(1) };
    }

    // Accumulate `c += buf[ki] * a[ki, :]`.
    let mut ki = 0usize;
    let out_end = unsafe { c.add(n) };

    while ki < n {
        let c_scalar = unsafe { *buf.add(ki) };
        let mut a_ptr = unsafe { a.add(ki * lda) };
        let mut out_ptr = c;

        while out_ptr != out_end {
            unsafe { *out_ptr += c_scalar * *a_ptr };
            out_ptr = unsafe { out_ptr.add(1) };
            a_ptr = unsafe { a_ptr.add(1) };
        }

        ki += 1;
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn muladd_r1n_rnn_1n_avx512(
    c: *mut f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    buf: *mut f32,
    n: usize,
) -> () {
    // Copy `c` into `buf`.
    let mut c_src = c as *const f32;
    let mut c_dst = buf;
    let c_src_vec_end = unsafe { c_src.add(n & !15) };
    let c_src_end = unsafe { c_src.add(n) };
    while c_src != c_src_vec_end {
        let cv = unsafe { _mm512_loadu_ps(c_src) };
        unsafe { _mm512_storeu_ps(c_dst, cv) };
        c_src = unsafe { c_src.add(16) };
        c_dst = unsafe { c_dst.add(16) };
    }
    while c_src != c_src_end {
        unsafe { *c_dst = *c_src };
        c_src = unsafe { c_src.add(1) };
        c_dst = unsafe { c_dst.add(1) };
    }

    // Load `b` into `c`.
    let mut c_ptr = c;
    let mut b_ptr = b;
    let c_end = unsafe { c.add(n) };
    let c_vec_end = unsafe { c.add(n & !15) };
    while c_ptr != c_vec_end {
        let bv = unsafe { _mm512_loadu_ps(b_ptr) };
        unsafe { _mm512_storeu_ps(c_ptr, bv) };
        c_ptr = unsafe { c_ptr.add(16) };
        b_ptr = unsafe { b_ptr.add(16) };
    }
    while c_ptr != c_end {
        unsafe { *c_ptr = *b_ptr };
        c_ptr = unsafe { c_ptr.add(1) };
        b_ptr = unsafe { b_ptr.add(1) };
    }

    // Accumulate `c += buf[ki] * a[ki, :]`.
    let mut ki = 0usize;
    let out_vec_end_base = unsafe { c.add(n & !15) };
    let out_end_base = unsafe { c.add(n) };
    while ki < n {
        let c_scalar = unsafe { *buf.add(ki) };
        let c_vec = _mm512_set1_ps(c_scalar);

        let mut a_ptr = unsafe { a.add(ki * lda) };
        let mut out_ptr = c;
        while out_ptr != out_vec_end_base {
            let out_v = unsafe { _mm512_loadu_ps(out_ptr) };
            let a_v = unsafe { _mm512_loadu_ps(a_ptr) };
            let sum_v = _mm512_fmadd_ps(c_vec, a_v, out_v);
            unsafe { _mm512_storeu_ps(out_ptr, sum_v) };

            out_ptr = unsafe { out_ptr.add(16) };
            a_ptr = unsafe { a_ptr.add(16) };
        }

        while out_ptr != out_end_base {
            unsafe { *out_ptr += c_scalar * *a_ptr };
            out_ptr = unsafe { out_ptr.add(1) };
            a_ptr = unsafe { a_ptr.add(1) };
        }

        ki += 1;
    }
}

/// Computes `C = C * A + B` for `m` rows.
///
/// # Parameters
///
/// - `c`: `C` base pointer, shape `(m, n)`.
/// - `rmc`: `true` if `C` is row-major.
/// - `ldc`: `C` leading dimension.
/// - `a`: `A` base pointer, shape `(n, n)`.
/// - `rma`: `true` if `A` is row-major.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(1, n)`.
/// - `buf`: Temporary storage for `n` `f32` values.
/// - `m`: Row count of `C`.
/// - `n`: Column count of `C` and `B`, and both dimensions of `A`.
///
/// # Safety
///
/// - `c`, `a`, and `b` cover the required ranges for `m`, `n`, layout flags, and strides.
/// - `buf` covers at least `n` `f32` values and is 64-byte aligned.
/// - `c`, `a`, and `b` do not overlap.
/// - `c`, `a`, and `b` are 4-byte aligned.
pub unsafe fn muladd_mn_nn_1n(
    c: *mut f32,
    rmc: bool,
    ldc: usize,
    a: *const f32,
    rma: bool,
    lda: usize,
    b: *const f32,
    buf: *mut f32,
    m: usize,
    n: usize,
) -> () {
    match (rmc, rma) {
        (true, true) => {
            let mut c_row = c;
            let c_row_end = unsafe { c.add(m * ldc) };
            if is_x86_feature_detected!("avx512f") {
                while c_row != c_row_end {
                    unsafe { muladd_r1n_rnn_1n_avx512(c_row, a, lda, b, buf, n) };
                    c_row = unsafe { c_row.add(ldc) };
                }
            } else {
                while c_row != c_row_end {
                    unsafe { muladd_r1n_rnn_1n_scalar(c_row, a, lda, b, buf, n) };
                    c_row = unsafe { c_row.add(ldc) };
                }
            }
        }
        _ => unimplemented!(),
    }
}
