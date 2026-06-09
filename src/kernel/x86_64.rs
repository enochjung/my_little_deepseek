//! x86_64 kernels for vector and matrix primitives.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn argmax_n_avx512(x: *const f32, n: usize) -> u32 {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;

    let mut v_max = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut v_max_idx = _mm512_setzero_si512();
    let mut v_curr_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let v_step = _mm512_set1_epi32(16);

    while ptr != vec_end {
        let xv = unsafe { _mm512_loadu_ps(ptr) };
        let mask = _mm512_cmp_ps_mask(xv, v_max, _CMP_GT_OQ);
        v_max = _mm512_mask_blend_ps(mask, v_max, xv);
        v_max_idx = _mm512_mask_blend_epi32(mask, v_max_idx, v_curr_idx);
        v_curr_idx = _mm512_add_epi32(v_curr_idx, v_step);
        ptr = unsafe { ptr.add(16) };
    }

    let mut tmp_max: [f32; 16] = [0.0; 16];
    let mut tmp_idx: [u32; 16] = [0; 16];

    unsafe { _mm512_storeu_ps(tmp_max.as_mut_ptr(), v_max) };
    unsafe { _mm512_storeu_si512(tmp_idx.as_mut_ptr() as *mut _, v_max_idx) };

    let mut global_max = f32::NEG_INFINITY;
    let mut global_max_idx = 0;

    if n >= 16 {
        for i in 0..16 {
            if tmp_max[i] > global_max {
                global_max = tmp_max[i];
                global_max_idx = tmp_idx[i];
            } else if tmp_max[i] == global_max && tmp_idx[i] < global_max_idx {
                global_max_idx = tmp_idx[i];
            }
        }
    }

    let mut curr_idx = (n & !15) as u32;
    while ptr != end {
        let v = unsafe { *ptr };
        if v > global_max {
            global_max = v;
            global_max_idx = curr_idx;
        }
        ptr = unsafe { ptr.add(1) };
        curr_idx += 1;
    }

    global_max_idx
}

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
pub(crate) unsafe fn safe_softmax_with_masking_n_avx512(
    x: *mut f32,
    alpha: f32,
    n_mask: usize,
    n: usize,
) -> () {
    let active_n = n.saturating_sub(n_mask);
    let end = unsafe { x.add(n) };

    if active_n == 0 {
        let mut ptr = x;
        while ptr != end {
            unsafe { *ptr = 0.0 };
            ptr = unsafe { ptr.add(1) };
        }
        return;
    }

    let vec_end = unsafe { x.add(active_n & !15) };
    let active_end = unsafe { x.add(active_n) };

    let mut v_max = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut ptr = x;
    while ptr != vec_end {
        let v = unsafe { _mm512_loadu_ps(ptr) };
        v_max = _mm512_max_ps(v_max, v);
        ptr = unsafe { ptr.add(16) };
    }

    let mut tmp_max: [f32; 16] = [0.0; 16];
    unsafe { _mm512_storeu_ps(tmp_max.as_mut_ptr(), v_max) };
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..16 {
        if tmp_max[i] > max_val {
            max_val = tmp_max[i];
        }
    }

    while ptr != active_end {
        let v = unsafe { *ptr };
        if v > max_val {
            max_val = v;
        }
        ptr = unsafe { ptr.add(1) };
    }

    let mut sum_vec = _mm512_setzero_ps();
    let v_max_vec = _mm512_set1_ps(max_val);
    let v_alpha = _mm512_set1_ps(alpha);
    let mut ptr = x;

    while ptr != vec_end {
        let v = unsafe { _mm512_loadu_ps(ptr) };
        let v_scaled = _mm512_mul_ps(_mm512_sub_ps(v, v_max_vec), v_alpha);

        let mut tmp: [f32; 16] = [0.0; 16];
        unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), v_scaled) };
        for i in 0..16 {
            tmp[i] = tmp[i].exp();
        }

        let exp_vec = unsafe { _mm512_loadu_ps(tmp.as_ptr()) };
        unsafe { _mm512_storeu_ps(ptr, exp_vec) };
        sum_vec = _mm512_add_ps(sum_vec, exp_vec);

        ptr = unsafe { ptr.add(16) };
    }

    let mut tmp_sum: [f32; 16] = [0.0; 16];
    unsafe { _mm512_storeu_ps(tmp_sum.as_mut_ptr(), sum_vec) };
    let mut sum = 0.0;
    for i in 0..16 {
        sum += tmp_sum[i];
    }

    while ptr != active_end {
        let v = unsafe { *ptr };
        let exp_v = (alpha * (v - max_val)).exp();
        unsafe { *ptr = exp_v };
        sum += exp_v;
        ptr = unsafe { ptr.add(1) };
    }

    let multiplier = 1.0 / sum;
    let multiplier_vec = _mm512_set1_ps(multiplier);
    let mut ptr = x;

    while ptr != vec_end {
        let exp_vec = unsafe { _mm512_loadu_ps(ptr) };
        let norm_vec = _mm512_mul_ps(exp_vec, multiplier_vec);
        unsafe { _mm512_storeu_ps(ptr, norm_vec) };
        ptr = unsafe { ptr.add(16) };
    }

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

        let out_v = unsafe { _mm512_loadu_ps(tmp.as_ptr()) };
        unsafe { _mm512_storeu_ps(ptr, out_v) };

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

        let out_v = unsafe { _mm512_loadu_ps(tmp.as_ptr()) };
        unsafe { _mm512_storeu_ps(ptr, out_v) };

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

const BLOCK_SIZE: usize = 64;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn mul_rmn_rmk_rkn_avx512(
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
                        let a_vec = _mm512_set1_ps(a_val);

                        let mut j = bj;
                        while j + 15 < j_end {
                            let c_ptr = unsafe { c.add(i * ldc as usize + j) };
                            let b_ptr = unsafe { b.add(k_idx * ldb as usize + j) };

                            let c_vec = unsafe { _mm512_loadu_ps(c_ptr) };
                            let b_vec = unsafe { _mm512_loadu_ps(b_ptr) };

                            let out = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                            unsafe { _mm512_storeu_ps(c_ptr, out) };

                            j += 16;
                        }

                        while j < j_end {
                            let b_val = unsafe { *b.add(k_idx * ldb as usize + j) };
                            unsafe { *c.add(i * ldc as usize + j) += a_val * b_val };
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn mul_rmn_rmk_ckn_avx512(
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        let mut sum_vec = _mm512_setzero_ps();
                        let mut k_idx = bk;

                        while k_idx + 15 < k_end {
                            let a_vec = unsafe { _mm512_loadu_ps(a.add(i * lda as usize + k_idx)) };
                            let b_vec = unsafe { _mm512_loadu_ps(b.add(k_idx + j * ldb as usize)) };
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                            k_idx += 16;
                        }

                        let mut sum = if bk == 0 {
                            0.0
                        } else {
                            unsafe { *c.add(i * ldc as usize + j) }
                        };
                        sum += _mm512_reduce_add_ps(sum_vec);

                        while k_idx < k_end {
                            unsafe {
                                sum += (*a.add(i * lda as usize + k_idx))
                                    * (*b.add(k_idx + j * ldb as usize))
                            };
                            k_idx += 1;
                        }

                        unsafe { *c.add(i * ldc as usize + j) = sum };
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn mul_rmn_rmk_ckn_r1n_avx512(
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        let mut sum_vec = _mm512_setzero_ps();
                        let mut k_idx = bk;

                        while k_idx + 15 < k_end {
                            let a_vec = unsafe { _mm512_loadu_ps(a.add(i * lda as usize + k_idx)) };
                            let b_vec = unsafe { _mm512_loadu_ps(b.add(k_idx + j * ldb as usize)) };
                            sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                            k_idx += 16;
                        }

                        let mut sum = if bk == 0 {
                            unsafe { *c.add(j) }
                        } else {
                            unsafe { *d.add(i * ldd as usize + j) }
                        };

                        sum += _mm512_reduce_add_ps(sum_vec);

                        while k_idx < k_end {
                            unsafe {
                                sum += (*a.add(i * lda as usize + k_idx))
                                    * (*b.add(k_idx + j * ldb as usize))
                            };
                            k_idx += 1;
                        }

                        unsafe { *d.add(i * ldd as usize + j) = sum };
                    }
                }
            }
        }
    }
}
