mod x86_64;

/// Copies `len` bytes from `src` to `dst` without allowing overlap.
///
/// # Safety
///
/// - `src` must be valid for reads of `len` bytes.
/// - `dst` must be valid for writes of `len` bytes.
/// - Source and destination ranges must not overlap.
pub(crate) unsafe fn copy(dst: *mut (), src: *const (), len: usize) -> () {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
}

/// Casts `n` BF16 values from `src` into `dst` as `f32`.
///
/// # Safety
///
/// - `dst` must be valid for writes of `n` contiguous `f32` values.
/// - `src` must be valid for reads of `n * 2` contiguous bytes.
/// - `dst` and `src` must not overlap.
/// - `dst` must be 64-byte aligned.
/// - `src` may be unaligned.
pub(crate) unsafe fn cast_bf16_to_f32_n_n(dst: *mut f32, src: *const (), n: usize) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::cast_bf16_to_f32_n_n_avx512(dst, src, n) };
        }
    }

    let dst_end = unsafe { dst.add(n) };
    let mut dst = dst;
    let mut src = src;

    while dst != dst_end {
        let bf16_u16 = unsafe { (src as *const u16).read_unaligned() } as u32;
        unsafe { *dst = f32::from_bits(bf16_u16 << 16) };
        dst = unsafe { dst.add(1) };
        src = unsafe { src.byte_add(2) };
    }
}

/// Computes RMS of `x`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `n > 0`.
pub(crate) unsafe fn rms_n(x: *const f32, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::rms_n_avx512(x, n) };
        }
    }

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

/// Scales `y` by `alpha * x` element-wise: `y[i] *= alpha * x[i]`
///
/// # Safety
///
/// - `x` and `y` must be valid for `n` contiguous `f32` values.
/// - `x` and `y` must not overlap in memory (restrict-like requirement).
/// - `x` and `y` must be aligned to `align_of::<f32>()` (4 bytes).
pub(crate) unsafe fn mul_n_n(y: *mut f32, x: *const f32, alpha: f32, n: usize) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::mul_n_n_avx512(y, x, alpha, n) };
        }
    }

    let x_end = unsafe { x.add(n) };
    let mut y = y;
    let mut x = x;

    while x != x_end {
        let yv = unsafe { *y };
        let xv = unsafe { *x };
        unsafe { *y = yv * xv * alpha };
        y = unsafe { y.add(1) };
        x = unsafe { x.add(1) };
    }
}

/// Adds `x` into `y` element-wise: `y[i] += x[i]`.
///
/// # Safety
///
/// - `x` and `y` must be valid for `n` contiguous `f32` values.
/// - `x` and `y` must not overlap in memory (restrict-like requirement).
/// - `x` and `y` must be aligned to `align_of::<f32>()` (4 bytes).
pub(crate) unsafe fn add_n_n(y: *mut f32, x: *const f32, n: usize) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::add_n_n_avx512(y, x, n) };
        }
    }

    let x_end = unsafe { x.add(n) };
    let mut y = y;
    let mut x = x;

    while x != x_end {
        unsafe { *y += *x };
        y = unsafe { y.add(1) };
        x = unsafe { x.add(1) };
    }
}

/// Applies SiLU in place: `x[i] = x[i] / (1 + exp(-x[i]))`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
pub(crate) unsafe fn silu_n(x: *mut f32, n: usize) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::silu_n_avx512(x, n) };
        }
    }

    let end = unsafe { x.add(n) };
    let mut ptr = x;

    while ptr != end {
        let v = unsafe { *ptr };
        unsafe { *ptr = v / (1.0 + (-v).exp()) };
        ptr = unsafe { ptr.add(1) };
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
pub(crate) unsafe fn muladd_1n_nn_1n(
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
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    return unsafe { x86_64::muladd_r1n_rnn_1n_avx512(c, a, lda, b, buf, n) };
                }
            }

            return unsafe { muladd_r1n_rnn_1n(c, a, lda, b, buf, n) };
        }
        _ => unimplemented!(),
    }
}

unsafe fn muladd_r1n_rnn_1n(
    c: *mut f32,
    a: *const f32,
    lda: usize,
    b: *const f32,
    buf: *mut f32,
    n: usize,
) -> () {
    // Copy `c` into `buf`.
    let c_src_end = unsafe { c.add(n) };
    let mut c_src = c as *const f32;
    let mut c_dst = buf;

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
pub(crate) unsafe fn muladd_mn_nn_1n(
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

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    while c_row != c_row_end {
                        unsafe { x86_64::muladd_r1n_rnn_1n_avx512(c_row, a, lda, b, buf, n) };
                        c_row = unsafe { c_row.add(ldc) };
                    }
                    return;
                }
            }
            {
                while c_row != c_row_end {
                    unsafe { muladd_r1n_rnn_1n(c_row, a, lda, b, buf, n) };
                    c_row = unsafe { c_row.add(ldc) };
                }
            }
        }
        _ => unimplemented!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case01_rms_n5() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let actual = unsafe { rms_n(x.as_ptr(), x.len()) };
        let expected = (11.0f32).sqrt();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn case02_rms_n16() {
        let x: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let actual = unsafe { rms_n(x.as_ptr(), x.len()) };
        let expected = (93.5f32).sqrt();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn case03_mul_n5() {
        let x = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let mut y = [1.0f32, 2.0, 3.0, 4.0, 5.0];

        unsafe { mul_n_n(y.as_mut_ptr(), x.as_ptr(), 0.5, x.len()) };

        let expected = [1.0f32, 4.0, 9.0, 16.0, 25.0];
        for i in 0..y.len() {
            assert!((y[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn case04_mul_n16() {
        let x: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut y = [1.0f32; 16];

        unsafe { mul_n_n(y.as_mut_ptr(), x.as_ptr(), 2.0, x.len()) };

        let expected: [f32; 16] = [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0,
        ];
        for i in 0..y.len() {
            assert!((y[i] - expected[i]).abs() < 1e-6);
        }
    }
}
