mod x86_64;

/// Returns the index of the maximum value in `x[0..n]`.
/// Typically used in `lm_head` for greedy decoding.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `n > 0` to avoid undefined behavior.
pub(crate) unsafe fn argmax_n(x: *const f32, n: usize) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::argmax_n_avx512(x, n) };
        }
    }

    let mut max_val = unsafe { *x };
    let mut max_idx = 0;

    let mut ptr = unsafe { x.add(1) };
    let end = unsafe { x.add(n) };
    let mut idx = 1;

    while ptr != end {
        let val = unsafe { *ptr };
        if val > max_val {
            max_val = val;
            max_idx = idx;
        }
        ptr = unsafe { ptr.add(1) };
        idx += 1;
    }

    max_idx as u32
}

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

/// Applies Softmax in place: `x[i] = exp(x[i]) / sum(exp(x)) * alpha`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
pub(crate) unsafe fn softmax_n(x: *mut f32, alpha: f32, n: usize) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::softmax_n_avx512(x, alpha, n) };
        }
    }

    let end = unsafe { x.add(n) };
    let mut ptr = x;

    let mut sum = 0.0;
    while ptr != end {
        let v = unsafe { *ptr };
        let exp_v = v.exp();
        unsafe { *ptr = exp_v };
        sum += exp_v;
        ptr = unsafe { ptr.add(1) };
    }

    let mut ptr = x;
    let multiplier = alpha / sum;
    while ptr != end {
        unsafe { *ptr *= multiplier };
        ptr = unsafe { ptr.add(1) };
    }
}

/// Fills `x[0..n]` with RoPE cosine factors computed as:
/// `cos(k / (theta^(2 * i / d)))` for `i` in `0..n`.
///
/// # Safety
///
/// - `x` must be valid for writes of `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `theta` and `d` should be positive to avoid undefined behavior in the power.
pub(crate) unsafe fn rope_cos_n(x: *mut f32, n: usize, k: f32, theta: f32, d: f32) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::rope_cos_n_avx512(x, n, k, theta, d) };
        }
    }

    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut i: usize = 0;

    while ptr != end {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *ptr = angle.cos() };
        ptr = unsafe { ptr.add(1) };
        i += 1;
    }
}

/// Fills `x[0..n]` with RoPE sine factors computed as:
/// `sin(k / (theta^(2 * i / d)))` for `i` in `0..n`.
///
/// # Safety
///
/// - `x` must be valid for writes of `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `theta` and `d` should be positive to avoid undefined behavior in the power.
pub(crate) unsafe fn rope_sin_n(x: *mut f32, n: usize, k: f32, theta: f32, d: f32) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::rope_sin_n_avx512(x, n, k, theta, d) };
        }
    }

    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut i: usize = 0;

    while ptr != end {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *ptr = angle.sin() };
        ptr = unsafe { ptr.add(1) };
        i += 1;
    }
}

/// Computes `C = A * B`
///
/// # Parameters
///
/// - `c`: `C` base pointer, shape `(m, n)`.
/// - `rmc`: `true` if `C` is row-major.
/// - `ldc`: `C` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `rma`: `true` if `A` is row-major.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
/// - `rmb`: `true` if `B` is row-major.
/// - `ldb`: `B` leading dimension.
/// - `m`: shape `m`
/// - `k`: shape `k`.
/// - `n`: shape `n`.
///
/// # Safety
///
/// - `a`, `b`, and `c` cover the required ranges for `m`, `k`, `n`, layout flags, and strides.
/// - `a`, `b`, and `c` do not overlap.
/// - `a`, `b`, and `c` are 4-byte aligned.
pub(crate) unsafe fn mul_mk_kn(
    c: *mut f32,
    rmc: bool,
    ldc: usize,
    a: *const f32,
    rma: bool,
    lda: usize,
    b: *const f32,
    rmb: bool,
    ldb: usize,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    todo!()
}

/// Computes `D = A * B + c` with `c` broadcasted.
///
/// # Parameters
///
/// - `d`: `D` base pointer, shape `(m, n)`.
/// - `rmd`: `true` if `D` is row-major.
/// - `ldd`: `D` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `rma`: `true` if `A` is row-major.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
/// - `rmb`: `true` if `B` is row-major.
/// - `ldb`: `B` leading dimension.
/// - `c`: `c` base pointer, shape `(1, n)`.
/// - `buf`: Temporary storage for `n` `f32` values.
/// - `m`: shape `m`
/// - `k`: shape `k`.
/// - `n`: shape `n`.
///
/// # Safety
///
/// - `a`, `b`, `c`, and `d` cover the required ranges for `m`, `k`, `n`, layout flags, and strides.
/// - `buf` covers at least `n` `f32` values and is 64-byte aligned.
/// - `a`, `b`, `c`, and `d` do not overlap.
/// - `a`, `b`, `c`, and `d` are 4-byte aligned.
pub(crate) unsafe fn muladd_mk_kn_1n(
    d: *mut f32,
    rmd: bool,
    ldd: usize,
    a: *const f32,
    rma: bool,
    lda: usize,
    b: *const f32,
    rmb: bool,
    ldb: usize,
    c: *const f32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    match (rmd, rma, rmb) {
        (true, true, true) => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    return unsafe {
                        x86_64::muladd_rmk_rkn_r1n_avx512(d, ldd, a, lda, b, ldb, c, m, k, n)
                    };
                }
            }
        }
        _ => {}
    }

    todo!()
    /*
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

    while ki < k {
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
    */
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

    #[test]
    fn case05_rope_cos_n5() {
        let mut buf = [0.0f32; 5];
        unsafe { rope_cos_n(buf.as_mut_ptr(), 5, 3.0, 10000.0, 128.0) };

        for i in 0..5usize {
            let angle = 3.0f32 / 10000.0f32.powf(2.0 * (i as f32) / 128.0);
            let expected = angle.cos();
            assert!((buf[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn case06_rope_sin_n5() {
        let mut buf = [0.0f32; 5];
        unsafe { rope_sin_n(buf.as_mut_ptr(), 5, 3.0, 10000.0, 128.0) };

        for i in 0..5usize {
            let angle = 3.0f32 / 10000.0f32.powf(2.0 * (i as f32) / 128.0);
            let expected = angle.sin();
            assert!((buf[i] - expected).abs() < 1e-6);
        }
    }
}
