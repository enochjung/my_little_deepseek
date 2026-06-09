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

/// Applies Safe Softmax in place with masking: `x[i] = exp(alpha * (x[i] - max(x))) / sum(exp(alpha * (x - max(x))))`.
/// Elements from `n - n_mask` to `n` are masked and set to `0.0`.
///
/// # Safety
///
/// - `x` must be valid for `n` contiguous `f32` values.
/// - `x` must be aligned to `align_of::<f32>()` (4 bytes).
/// - `alpha` must be a positive number.
pub(crate) unsafe fn safe_softmax_with_masking_n(
    x: *mut f32,
    alpha: f32,
    n_mask: usize,
    n: usize,
) -> () {
    let active_n = n.saturating_sub(n_mask);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { safe_softmax_with_masking_n_avx512(x, alpha, n_mask, n) };
        }
    }

    let end = unsafe { x.add(n) };

    if active_n == 0 {
        let mut ptr = x;
        while ptr != end {
            unsafe { *ptr = 0.0 };
            ptr = unsafe { ptr.add(1) };
        }
        return;
    }

    let active_end = unsafe { x.add(active_n) };

    let mut max_val = f32::NEG_INFINITY;
    let mut ptr = x;
    while ptr != active_end {
        let v = unsafe { *ptr };
        if v > max_val {
            max_val = v;
        }
        ptr = unsafe { ptr.add(1) };
    }

    let mut sum = 0.0;
    let mut ptr = x;
    while ptr != active_end {
        let v = unsafe { *ptr };
        let exp_v = (alpha * (v - max_val)).exp();
        unsafe { *ptr = exp_v };
        sum += exp_v;
        ptr = unsafe { ptr.add(1) };
    }

    let multiplier = 1.0 / sum;
    let mut ptr = x;
    while ptr != active_end {
        unsafe { *ptr *= multiplier };
        ptr = unsafe { ptr.add(1) };
    }

    let mut ptr = active_end;
    while ptr != end {
        unsafe { *ptr = 0.0 };
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

const BLOCK_SIZE: usize = 64;

/// Computes `C = A * B`
///
/// # Parameters
///
/// - `c`: `C` base pointer, shape `(m, n)`.
/// - `ldc`: `C` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
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
pub(crate) unsafe fn mul_rmn_rmk_rkn(
    c: *mut f32,
    ldc: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::mul_rmn_rmk_rkn_avx512(c, ldc, a, lda, b, ldb, m, k, n) };
        }
    }

    for i in 0..m {
        for j in 0..n {
            unsafe { *c.add(i * ldc as usize + j) = 0.0 };
        }
    }

    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bk in (0..k).step_by(BLOCK_SIZE) {
            for bj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let k_end = (bk + BLOCK_SIZE).min(k);
                let j_end = (bj + BLOCK_SIZE).min(n);

                for i in bi..i_end {
                    for k_idx in bk..k_end {
                        let a_val = unsafe { *a.add(i * lda as usize + k_idx) };
                        for j in bj..j_end {
                            let b_val = unsafe { *b.add(k_idx * ldb as usize + j) };
                            unsafe { *c.add(i * ldc as usize + j) += a_val * b_val };
                        }
                    }
                }
            }
        }
    }
}

/// Computes `C = A * B`, but `B` is column-major.
///
/// # Parameters
///
/// - `c`: `C` base pointer, shape `(m, n)`.
/// - `ldc`: `C` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
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
pub(crate) unsafe fn mul_rmn_rmk_ckn(
    c: *mut f32,
    ldc: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { x86_64::mul_rmn_rmk_ckn_avx512(c, ldc, a, lda, b, ldb, m, k, n) };
        }
    }

    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        let mut sum = if bk == 0 {
                            0.0
                        } else {
                            unsafe { *c.add(i * ldc as usize + j) }
                        };
                        for k_idx in bk..k_end {
                            unsafe {
                                sum += (*a.add(i * lda as usize + k_idx))
                                    * (*b.add(k_idx + j * ldb as usize))
                            };
                        }
                        unsafe { *c.add(i * ldc as usize + j) = sum };
                    }
                }
            }
        }
    }
}

/// Computes `D = A * B + c` with `c` broadcasted along m-dimension.
///
/// # Parameters
///
/// - `d`: `D` base pointer, shape `(m, n)`.
/// - `ldd`: `D` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
/// - `ldb`: `B` leading dimension.
/// - `c`: `c` base pointer, shape `(1, n)`.
/// - `m`: shape `m`
/// - `k`: shape `k`.
/// - `n`: shape `n`.
///
/// # Safety
///
/// - `a`, `b`, `c`, and `d` cover the required ranges for `m`, `k`, `n`, layout flags, and strides.
/// - `a`, `b`, `c`, and `d` do not overlap.
/// - `a`, `b`, `c`, and `d` are 4-byte aligned.
pub(crate) unsafe fn mul_rmn_rmk_rkn_r1n(
    d: *mut f32,
    ldd: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    c: *const f32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe {
                x86_64::mul_rmn_rmk_rkn_r1n_avx512(d, ldd, a, lda, b, ldb, c, m, k, n)
            };
        }
    }

    for i in 0..m {
        for j in 0..n {
            unsafe { *d.add(i * ldd as usize + j) = *c.add(j) };
        }
    }

    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bk in (0..k).step_by(BLOCK_SIZE) {
            for bj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let k_end = (bk + BLOCK_SIZE).min(k);
                let j_end = (bj + BLOCK_SIZE).min(n);

                for i in bi..i_end {
                    for k_idx in bk..k_end {
                        let a_val = unsafe { *a.add(i * lda as usize + k_idx) };
                        for j in bj..j_end {
                            let b_val = unsafe { *b.add(k_idx * ldb as usize + j) };
                            unsafe { *d.add(i * ldd as usize + j) += a_val * b_val };
                        }
                    }
                }
            }
        }
    }
}

/// Computes `D = A * B + c` with `c` broadcasted along m-dimension, but `B` is column-major.
///
/// # Parameters
///
/// - `d`: `D` base pointer, shape `(m, n)`.
/// - `ldd`: `D` leading dimension.
/// - `a`: `A` base pointer, shape `(m, k)`.
/// - `lda`: `A` leading dimension.
/// - `b`: `B` base pointer, shape `(k, n)`.
/// - `ldb`: `B` leading dimension.
/// - `c`: `c` base pointer, shape `(1, n)`.
/// - `m`: shape `m`
/// - `k`: shape `k`.
/// - `n`: shape `n`.
///
/// # Safety
///
/// - `a`, `b`, `c`, and `d` cover the required ranges for `m`, `k`, `n`, layout flags, and strides.
/// - `a`, `b`, `c`, and `d` do not overlap.
/// - `a`, `b`, `c`, and `d` are 4-byte aligned.
pub(crate) unsafe fn mul_rmn_rmk_ckn_r1n(
    d: *mut f32,
    ldd: u32,
    a: *const f32,
    lda: u32,
    b: *const f32,
    ldb: u32,
    c: *const f32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe {
                x86_64::mul_rmn_rmk_ckn_r1n_avx512(d, ldd, a, lda, b, ldb, c, m, k, n)
            };
        }
    }

    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        let mut sum = if bk == 0 {
                            unsafe { *c.add(j) }
                        } else {
                            unsafe { *d.add(i * ldd as usize + j) }
                        };

                        for k_idx in bk..k_end {
                            unsafe {
                                sum += (*a.add(i * lda as usize + k_idx))
                                    * (*b.add(k_idx + j * ldb as usize))
                            };
                        }

                        unsafe { *d.add(i * ldd as usize + j) = sum };
                    }
                }
            }
        }
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
