//! Low-level `Cpu` device math operations.
//!
//! This module provides optimized SIMD and scalar fallback math routines
//! for model inference, including basic BLAS-like operations and activations.

mod x86_64;

/// Returns the index of the maximum value in a contiguous sequence of `f32`s.
///
/// This is typically used in the language model head (`lm_head`) for greedy decoding.
///
/// # Safety
///
/// * `x` must be valid for reading `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
/// * `n` must be greater than 0.
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

/// Copies `len` bytes from `src` to `dst`.
///
/// # Safety
///
/// * `src` must be valid for reading `len` contiguous bytes.
/// * `dst` must be valid for writing `len` contiguous bytes.
/// * `src` and `dst` must not overlap.
pub(crate) unsafe fn copy(dst: *mut (), src: *const (), len: usize) -> () {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
}

/// Casts `n` contiguous `bfloat16` elements to `f32`.
///
/// # Safety
///
/// * `src` must be valid for reading `n * 2` contiguous bytes.
/// * `dst` must be valid for writing `n` contiguous `f32` elements.
/// * `src` and `dst` must be properly aligned.
/// * `src` and `dst` must not overlap.
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

///
/// # Safety
///
/// * `x` must be valid for reading `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
/// * `n` must be greater than 0.
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

/// Scales elements of `y` by corresponding elements in `x` and a scalar `alpha`.
///
/// Performs the element-wise operation: `y[i] *= alpha * x[i]`.
///
/// # Safety
///
/// * `x` must be valid for reading `n` contiguous `f32` elements.
/// * `y` must be valid for reading and writing `n` contiguous `f32` elements.
/// * `x` and `y` must be properly aligned.
/// * `x` and `y` must not overlap.
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

/// Adds elements of `x` into `y`.
///
/// Performs the element-wise operation: `y[i] += x[i]`.
///
/// # Safety
///
/// * `x` must be valid for reading `n` contiguous `f32` elements.
/// * `y` must be valid for reading and writing `n` contiguous `f32` elements.
/// * `x` and `y` must be properly aligned.
/// * `x` and `y` must not overlap.
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

/// Applies the SiLU (Sigmoid Linear Unit) activation function in-place.
///
/// Evaluates `x[i] = x[i] / (1 + exp(-x[i]))` for each element.
///
/// # Safety
///
/// * `x` must be valid for reading and writing `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
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

/// Applies a numerically stable Softmax in-place with optional masking.
///
/// Elements in the range `[n - n_mask, n)` are masked and zeroed out.
///
/// # Safety
///
/// * `x` must be valid for reading and writing `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
/// * `alpha` must be positive.
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
            return unsafe { x86_64::safe_softmax_with_masking_n_avx512(x, alpha, n_mask, n) };
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

/// Populates the sequence with Rotary Positional Embedding (RoPE) cosine factors.
///
/// # Safety
///
/// * `x` must be valid for writing `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
/// * `theta` and `d` must be positive.
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

/// Populates the sequence with Rotary Positional Embedding (RoPE) sine factors.
///
/// # Safety
///
/// * `x` must be valid for writing `n` contiguous `f32` elements.
/// * `x` must be properly aligned.
/// * `theta` and `d` must be positive.
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

/// Computes the matrix multiplication `C = A * B` (all matrices row-major).
///
/// # Safety
///
/// * `a`, `b`, and `c` must be valid for reading and writing over the dimensions specified by `m`, `k`, `n` and their respective strides.
/// * `a`, `b`, and `c` must be properly aligned.
/// * `a`, `b`, and `c` must not overlap.
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

/// Computes the matrix multiplication `C = A * B` where `B` is column-major.
///
/// # Safety
///
/// * `a`, `b`, and `c` must be valid for reading and writing over the dimensions specified by `m`, `k`, `n` and their respective strides.
/// * `a`, `b`, and `c` must be properly aligned.
/// * `a`, `b`, and `c` must not overlap.
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

/// Computes the matrix multiplication `D = A * B + c` with a row-broadcasted bias.
///
/// The matrix `B` is assumed to be column-major. The bias `c` has shape `(1, n)`
/// and is broadcasted across the `m` dimension.
///
/// # Safety
///
/// * `a`, `b`, `c`, and `d` must be valid for reading and writing over the dimensions specified by `m`, `k`, `n` and their respective strides.
/// * `a`, `b`, `c`, and `d` must be properly aligned.
/// * `a`, `b`, `c`, and `d` must not overlap.
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

    fn assert(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "illegal test data");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "data[{}] mismatch: expected {}, actual {}",
                i,
                e,
                a,
            );
        }
    }

    #[test]
    fn argmax_n5() {
        let x = [1.0f32, 5.0, 3.0, 2.0, 4.0];
        let actual = unsafe { argmax_n(x.as_ptr(), x.len()) };
        assert_eq!(actual, 1);
    }

    #[test]
    fn argmax_n20() {
        let mut x = [0.0f32; 20];
        for i in 0..20 {
            x[i] = i as f32;
        }
        x[15] = 99.0;
        let actual = unsafe { argmax_n(x.as_ptr(), x.len()) };
        assert_eq!(actual, 15);
    }

    #[test]
    fn copy_n5() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut dst = [0.0f32; 5];
        unsafe {
            copy(
                dst.as_mut_ptr() as *mut (),
                src.as_ptr() as *const (),
                5 * std::mem::size_of::<f32>(),
            )
        };
        assert_eq!(src, dst);
    }

    #[test]
    fn copy_n20() {
        let mut src = [0.0f32; 20];
        for i in 0..20 {
            src[i] = i as f32;
        }
        let mut dst = [0.0f32; 20];
        unsafe {
            copy(
                dst.as_mut_ptr() as *mut (),
                src.as_ptr() as *const (),
                20 * std::mem::size_of::<f32>(),
            )
        };
        assert_eq!(src, dst);
    }

    #[test]
    fn cast_bf16_to_f32_n_n5() {
        let f32_vals = [1.0f32, -2.5, 3.3, 4.125, -5.0];
        let mut bf16_vals = [0u16; 5];
        for i in 0..5 {
            bf16_vals[i] = (f32_vals[i].to_bits() >> 16) as u16;
        }
        let mut dst = [0.0f32; 5];
        unsafe { cast_bf16_to_f32_n_n(dst.as_mut_ptr(), bf16_vals.as_ptr() as *const (), 5) };

        for i in 0..5 {
            let expected = f32::from_bits((bf16_vals[i] as u32) << 16);
            assert_eq!(dst[i], expected);
        }
    }

    #[test]
    fn cast_bf16_to_f32_n_n20() {
        let mut bf16_vals = [0u16; 20];
        for i in 0..20 {
            let val = i as f32 * 1.5;
            bf16_vals[i] = (val.to_bits() >> 16) as u16;
        }
        let mut dst = [0.0f32; 20];
        unsafe { cast_bf16_to_f32_n_n(dst.as_mut_ptr(), bf16_vals.as_ptr() as *const (), 20) };

        for i in 0..20 {
            let expected = f32::from_bits((bf16_vals[i] as u32) << 16);
            assert_eq!(dst[i], expected);
        }
    }

    #[test]
    fn rms_n5() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let actual = unsafe { rms_n(x.as_ptr(), x.len()) };
        let expected = (55.0f32 / 5.0).sqrt();
        assert!((actual - expected).abs() < 1e-5);
    }

    #[test]
    fn rms_n20() {
        let mut x = [0.0f32; 20];
        let mut sq_sum = 0.0;
        for i in 0..20 {
            let val = (i + 1) as f32;
            x[i] = val;
            sq_sum += val * val;
        }
        let actual = unsafe { rms_n(x.as_ptr(), x.len()) };
        let expected = (sq_sum / 20.0).sqrt();
        assert!((actual - expected).abs() < 1e-5);
    }

    #[test]
    fn mul_n_n5() {
        let x = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let mut y = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        unsafe { mul_n_n(y.as_mut_ptr(), x.as_ptr(), 0.5, 5) };

        let expected = [1.0f32, 4.0, 9.0, 16.0, 25.0];
        assert(&y, &expected);
    }

    #[test]
    fn mul_n_n20() {
        let mut x = [0.0f32; 20];
        let mut y = [1.0f32; 20];
        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            x[i] = (i + 1) as f32;
            expected[i] = x[i] * 2.0;
        }
        unsafe { mul_n_n(y.as_mut_ptr(), x.as_ptr(), 2.0, 20) };
        assert(&y, &expected);
    }

    #[test]
    fn add_n_n5() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut y = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        unsafe { add_n_n(y.as_mut_ptr(), x.as_ptr(), 5) };

        let expected = [6.0f32; 5];
        assert(&y, &expected);
    }

    #[test]
    fn add_n_n20() {
        let mut x = [0.0f32; 20];
        let mut y = [0.0f32; 20];
        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            x[i] = i as f32;
            y[i] = (i * 2) as f32;
            expected[i] = x[i] + y[i];
        }
        unsafe { add_n_n(y.as_mut_ptr(), x.as_ptr(), 20) };
        assert(&y, &expected);
    }

    #[test]
    fn silu_n5() {
        let mut x = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut expected = [0.0f32; 5];
        for i in 0..5 {
            expected[i] = x[i] / (1.0 + (-x[i]).exp());
        }
        unsafe { silu_n(x.as_mut_ptr(), 5) };
        assert(&x, &expected);
    }

    #[test]
    fn silu_n20() {
        let mut x = [0.0f32; 20];
        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            let val = (i as f32) - 10.0;
            x[i] = val;
            expected[i] = val / (1.0 + (-val).exp());
        }
        unsafe { silu_n(x.as_mut_ptr(), 20) };
        assert(&x, &expected);
    }

    #[test]
    fn safe_softmax_with_masking_n5() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let alpha = 1.0;
        let mut expected = [0.0f32; 5];
        let max_val = 3.0;

        let sum = (1.0f32 - max_val).exp() + (2.0f32 - max_val).exp() + (3.0f32 - max_val).exp();
        expected[0] = (1.0f32 - max_val).exp() / sum;
        expected[1] = (2.0f32 - max_val).exp() / sum;
        expected[2] = (3.0f32 - max_val).exp() / sum;

        unsafe { safe_softmax_with_masking_n(x.as_mut_ptr(), alpha, 2, 5) };
        assert(&x, &expected);
    }

    #[test]
    fn safe_softmax_with_masking_n20() {
        let mut x = [0.0f32; 20];
        for i in 0..20 {
            x[i] = i as f32;
        }
        let alpha = 0.5;
        let n_mask = 5;
        let max_val = 14.0;
        let mut expected = [0.0f32; 20];
        let mut sum = 0.0;

        for i in 0..15 {
            sum += (alpha * (i as f32 - max_val)).exp();
        }
        for i in 0..15 {
            expected[i] = (alpha * (i as f32 - max_val)).exp() / sum;
        }

        unsafe { safe_softmax_with_masking_n(x.as_mut_ptr(), alpha, n_mask, 20) };
        assert(&x, &expected);
    }

    #[test]
    fn rope_cos_n5() {
        let mut buf = [0.0f32; 5];
        let (k, theta, d) = (3.0, 10000.0, 128.0);
        unsafe { rope_cos_n(buf.as_mut_ptr(), 5, k, theta, d) };

        let mut expected = [0.0f32; 5];
        for i in 0..5 {
            expected[i] = (k / theta.powf(2.0 * (i as f32) / d)).cos();
        }
        assert(&buf, &expected);
    }

    #[test]
    fn rope_cos_n20() {
        let mut buf = [0.0f32; 20];
        let (k, theta, d) = (1.5, 10000.0, 64.0);
        unsafe { rope_cos_n(buf.as_mut_ptr(), 20, k, theta, d) };

        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            expected[i] = (k / theta.powf(2.0 * (i as f32) / d)).cos();
        }
        assert(&buf, &expected);
    }

    #[test]
    fn rope_sin_n5() {
        let mut buf = [0.0f32; 5];
        let (k, theta, d) = (3.0, 10000.0, 128.0);
        unsafe { rope_sin_n(buf.as_mut_ptr(), 5, k, theta, d) };

        let mut expected = [0.0f32; 5];
        for i in 0..5 {
            expected[i] = (k / theta.powf(2.0 * (i as f32) / d)).sin();
        }
        assert(&buf, &expected);
    }

    #[test]
    fn rope_sin_n20() {
        let mut buf = [0.0f32; 20];
        let (k, theta, d) = (1.5, 10000.0, 64.0);
        unsafe { rope_sin_n(buf.as_mut_ptr(), 20, k, theta, d) };

        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            expected[i] = (k / theta.powf(2.0 * (i as f32) / d)).sin();
        }
        assert(&buf, &expected);
    }

    #[test]
    fn mul_rmn_rmk_rkn_n5() {
        let (m, k, n) = (1, 2, 5);
        let a = [1.0f32, 2.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0.0f32; 5];
        unsafe {
            mul_rmn_rmk_rkn(
                c.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                n as u32,
                m,
                k,
                n,
            )
        };

        let expected = [5.0f32, 8.0, 11.0, 14.0, 17.0];
        assert(&c, &expected);
    }

    #[test]
    fn mul_rmn_rmk_rkn_n20() {
        let (m, k, n) = (1, 2, 20);
        let a = [1.0f32, 2.0];
        let mut b = [0.0f32; 40];
        for i in 0..20 {
            b[i] = i as f32;
            b[i + 20] = (i + 1) as f32;
        }
        let mut c = [0.0f32; 20];
        unsafe {
            mul_rmn_rmk_rkn(
                c.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                n as u32,
                m,
                k,
                n,
            )
        };

        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            expected[i] = 1.0 * b[i] + 2.0 * b[i + 20];
        }
        assert(&c, &expected);
    }

    #[test]
    fn mul_rmn_rmk_ckn_n5() {
        let (m, k, n) = (1, 2, 5);
        let a = [1.0f32, 2.0];
        let b = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0];
        let mut c = [0.0f32; 5];
        unsafe {
            mul_rmn_rmk_ckn(
                c.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                k as u32,
                m,
                k,
                n,
            )
        };

        let expected = [5.0f32, 8.0, 11.0, 14.0, 17.0];
        assert(&c, &expected);
    }

    #[test]
    fn mul_rmn_rmk_ckn_n20() {
        let (m, k, n) = (1, 2, 20);
        let a = [1.0f32, 2.0];
        let mut b = [0.0f32; 40];
        for i in 0..20 {
            b[i * 2] = i as f32;
            b[i * 2 + 1] = (i + 1) as f32;
        }
        let mut c = [0.0f32; 20];
        unsafe {
            mul_rmn_rmk_ckn(
                c.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                k as u32,
                m,
                k,
                n,
            )
        };

        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            expected[i] = 1.0 * b[i * 2] + 2.0 * b[i * 2 + 1];
        }
        assert(&c, &expected);
    }

    #[test]
    fn mul_rmn_rmk_ckn_r1n_n5() {
        let (m, k, n) = (1, 2, 5);
        let a = [1.0f32, 2.0];
        let b = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0];
        let bias = [0.5f32, 0.5, 0.5, 0.5, 0.5];
        let mut d = [0.0f32; 5];
        unsafe {
            mul_rmn_rmk_ckn_r1n(
                d.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                k as u32,
                bias.as_ptr(),
                m,
                k,
                n,
            )
        };

        let expected = [5.5f32, 8.5, 11.5, 14.5, 17.5];
        assert(&d, &expected);
    }

    #[test]
    fn mul_rmn_rmk_ckn_r1n_n20() {
        let (m, k, n) = (1, 2, 20);
        let a = [1.0f32, 2.0];
        let mut b = [0.0f32; 40];
        for i in 0..20 {
            b[i * 2] = i as f32;
            b[i * 2 + 1] = (i + 1) as f32;
        }
        let bias = [0.5f32; 20];
        let mut d = [0.0f32; 20];
        unsafe {
            mul_rmn_rmk_ckn_r1n(
                d.as_mut_ptr(),
                n as u32,
                a.as_ptr(),
                k as u32,
                b.as_ptr(),
                k as u32,
                bias.as_ptr(),
                m,
                k,
                n,
            )
        };

        let mut expected = [0.0f32; 20];
        for i in 0..20 {
            expected[i] = 1.0 * b[i * 2] + 2.0 * b[i * 2 + 1] + 0.5;
        }
        assert(&d, &expected);
    }
}
