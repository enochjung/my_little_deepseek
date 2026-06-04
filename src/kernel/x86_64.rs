//! x86_64 kernels for vector and matrix primitives.
//!
//! Layout:
//! - Row-major: `base[row * ld + col]`
//! - Column-major: `base[row + col * ld]`

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn cast_bf16_to_f32_n_n_avx512(dst: *mut f32, src: *const (), n: usize) -> () {
    let vec_end = unsafe { dst.add(n & !15) };
    let dst_end = unsafe { dst.add(n) };
    let mut dst = dst;
    let mut src = src;

    while dst != vec_end {
        let bf16_v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let bits_u32 = _mm512_cvtepu16_epi32(bf16_v);
        let bits_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(bits_u32, 16));
        unsafe { _mm512_store_ps(dst, bits_f32) };
        dst = unsafe { dst.add(16) };
        src = unsafe { src.byte_add(32) };
    }

    while dst != dst_end {
        let bf16_u16 = unsafe { (src as *const u16).read_unaligned() } as u32;
        unsafe { *dst = f32::from_bits(bf16_u16 << 16) };
        dst = unsafe { dst.add(1) };
        src = unsafe { src.byte_add(2) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn rms_n_avx512(x: *const f32, n: usize) -> f32 {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut acc = _mm512_setzero_ps();

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn mul_n_n_avx512(y: *mut f32, x: *const f32, alpha: f32, n: usize) -> () {
    let alpha_v = _mm512_set1_ps(alpha);
    let x_vec_end = unsafe { x.add(n & !15) };
    let x_end = unsafe { x.add(n) };
    let mut y = y;
    let mut x = x;

    while x != x_vec_end {
        let yv = unsafe { _mm512_loadu_ps(y as *const f32) };
        let xv = unsafe { _mm512_loadu_ps(x) };
        let out = _mm512_mul_ps(yv, _mm512_mul_ps(xv, alpha_v));
        unsafe { _mm512_storeu_ps(y, out) };
        y = unsafe { y.add(16) };
        x = unsafe { x.add(16) };
    }

    while x != x_end {
        let yv = unsafe { *y };
        let xv = unsafe { *x };
        unsafe { *y = yv * xv * alpha };
        y = unsafe { y.add(1) };
        x = unsafe { x.add(1) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn add_n_n_avx512(y: *mut f32, x: *const f32, n: usize) -> () {
    let x_vec_end = unsafe { x.add(n & !15) };
    let x_end = unsafe { x.add(n) };
    let mut y = y;
    let mut x = x;

    while x != x_vec_end {
        let yv = unsafe { _mm512_loadu_ps(y as *const f32) };
        let xv = unsafe { _mm512_loadu_ps(x) };
        let out = _mm512_add_ps(yv, xv);
        unsafe { _mm512_storeu_ps(y, out) };
        y = unsafe { y.add(16) };
        x = unsafe { x.add(16) };
    }

    while x != x_end {
        unsafe { *y += *x };
        y = unsafe { y.add(1) };
        x = unsafe { x.add(1) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn silu_n_avx512(x: *mut f32, n: usize) -> () {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;

    while ptr != vec_end {
        let mut tmp: [f32; 16] = [0.0; 16];
        let v = unsafe { _mm512_loadu_ps(ptr) };
        unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), v) };
        for i in 0..16 {
            let vi = tmp[i];
            tmp[i] = vi / (1.0 + (-vi).exp());
        }

        let out_v = unsafe { _mm512_loadu_ps(tmp.as_ptr()) };
        unsafe { _mm512_storeu_ps(ptr, out_v) };
        ptr = unsafe { ptr.add(16) };
    }

    while ptr != end {
        let v = unsafe { *ptr };
        unsafe { *ptr = v / (1.0 + (-v).exp()) };
        ptr = unsafe { ptr.add(1) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn rope_cos_n_avx512(x: *mut f32, n: usize, k: f32, theta: f32, d: f32) -> () {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut i: usize = 0;

    while ptr != vec_end {
        let mut tmp: [f32; 16] = [0.0; 16];
        for j in 0..16 {
            let idx = i + j;
            let angle = k / theta.powf(2.0 * (idx as f32) / d);
            tmp[j] = angle.cos();
        }

        let out_v = _mm512_loadu_ps(tmp.as_ptr());
        _mm512_storeu_ps(ptr, out_v);

        ptr = unsafe { ptr.add(16) };
        i += 16;
    }

    while ptr != end {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *ptr = angle.cos() };
        ptr = unsafe { ptr.add(1) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn rope_sin_n_avx512(x: *mut f32, n: usize, k: f32, theta: f32, d: f32) -> () {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut i: usize = 0;

    while ptr != vec_end {
        let mut tmp: [f32; 16] = [0.0; 16];
        for j in 0..16 {
            let idx = i + j;
            let angle = k / theta.powf(2.0 * (idx as f32) / d);
            tmp[j] = angle.sin();
        }

        let out_v = _mm512_loadu_ps(tmp.as_ptr());
        _mm512_storeu_ps(ptr, out_v);

        ptr = unsafe { ptr.add(16) };
        i += 16;
    }

    while ptr != end {
        let angle = k / theta.powf(2.0 * (i as f32) / d);
        unsafe { *ptr = angle.sin() };
        ptr = unsafe { ptr.add(1) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn muladd_rmk_rkn_r1n_avx512(
    d: *mut f32,
    ldd: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *const f32,
    buf: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> () {
    todo!()
    /*
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
    while ki < k {
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
    */
}
