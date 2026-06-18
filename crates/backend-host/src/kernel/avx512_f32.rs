//! AVX-512 kernels for f32
//! NOT TESTED

#[cfg(target_feature = "avx512f")]
use std::arch::x86_64::*;

#[cfg(target_feature = "avx512f")]
const BLOCK_SIZE: usize = 64;

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_add_assign(y: *mut f32, x: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            let xv = _mm512_loadu_ps(x.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_add_ps(yv, xv));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) += *x.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_add_assign_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    let mut y = y;
    let mut a = a;

    let vec_end = n & !15;
    for _ in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let yv = _mm512_loadu_ps(y.add(j));
                let av = _mm512_loadu_ps(a.add(j));
                _mm512_storeu_ps(y.add(j), _mm512_add_ps(yv, av));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(j) += *a.add(j) };
            j += 1;
        }

        y = unsafe { y.add(ldy as usize) };
        a = unsafe { a.add(lda as usize) };
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_add_assign_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    let mut y = y;
    let mut a = a;

    let vec_end = n & !15;
    for _ in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize);
                }
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let yv = _mm512_loadu_ps(y.add(j));
                _mm512_storeu_ps(y.add(j), _mm512_add_ps(yv, av));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(j) += *a.add(j * lda as usize) };
            j += 1;
        }

        y = unsafe { y.add(ldy as usize) };
        a = unsafe { a.add(1) };
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn argmax(x: *const f32, n: usize) -> u32 {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;

    let mut v_max = unsafe { _mm512_set1_ps(f32::NEG_INFINITY) };
    let mut v_max_idx = unsafe { _mm512_setzero_si512() };
    let mut v_curr_idx =
        unsafe { _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) };
    let v_step = unsafe { _mm512_set1_epi32(16) };

    while ptr != vec_end {
        unsafe {
            let xv = _mm512_loadu_ps(ptr);
            let mask = _mm512_cmp_ps_mask(xv, v_max, _CMP_GT_OQ);
            v_max = _mm512_mask_blend_ps(mask, v_max, xv);
            v_max_idx = _mm512_mask_blend_epi32(mask, v_max_idx, v_curr_idx);
            v_curr_idx = _mm512_add_epi32(v_curr_idx, v_step);
            ptr = ptr.add(16);
        }
    }

    let mut tmp_max: [f32; 16] = [0.0; 16];
    let mut tmp_idx: [u32; 16] = [0; 16];

    unsafe {
        _mm512_storeu_ps(tmp_max.as_mut_ptr(), v_max);
        _mm512_storeu_si512(tmp_idx.as_mut_ptr() as *mut _, v_max_idx);
    }

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
        unsafe {
            let v = *ptr;
            if v > global_max {
                global_max = v;
                global_max_idx = curr_idx;
            }
            ptr = ptr.add(1);
        }
        curr_idx += 1;
    }

    global_max_idx
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn copy(y: *mut f32, x: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe { _mm512_storeu_ps(y.add(i), _mm512_loadu_ps(x.add(i))) };
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) = *x.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn copy_rmn_rmn(y: *mut f32, ldy: u32, a: *const f32, lda: u32, m: usize, n: usize) {
    let mut y = y;
    let mut a = a;

    let vec_end = n & !15;
    for _ in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                _mm512_storeu_ps(y.add(j), _mm512_loadu_ps(a.add(j)));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(j) = *a.add(j) };
            j += 1;
        }

        y = unsafe { y.add(ldy as usize) };
        a = unsafe { a.add(lda as usize) };
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn copy_rmn_cmn(y: *mut f32, ldy: u32, a: *const f32, lda: u32, m: usize, n: usize) {
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp = [0.0; 16];
                for k in 0..16 {
                    tmp[k] = *a.add((j + k) * lda as usize + i);
                }
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_loadu_ps(tmp.as_ptr()));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(i * ldy as usize + j) = *a.add(j * lda as usize + i) };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn fill(y: *mut f32, value: f32, n: usize) {
    let vec_end = n & !15;
    let v = unsafe { _mm512_set1_ps(value) };
    let mut i = 0;
    while i < vec_end {
        unsafe { _mm512_storeu_ps(y.add(i), v) };
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) = value };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn scalar_mul_assign(y: *mut f32, value: f32, n: usize) {
    let vec_end = n & !15;
    let v = unsafe { _mm512_set1_ps(value) };
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_mul_ps(yv, v));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) *= value };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_mul_assign(y: *mut f32, x: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            let xv = _mm512_loadu_ps(x.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_mul_ps(yv, xv));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) *= *x.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_mul_assign_rmn_rmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(yv, av));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(i * lda as usize + j) };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_mul_assign_rmn_cmn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(yv, av));
            }
            j += 16;
        }
        while j < n {
            unsafe { *y.add(i * ldy as usize + j) *= *a.add(j * lda as usize + i) };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_mul(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let av = _mm512_loadu_ps(a.add(i));
            let bv = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_mul_ps(av, bv));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) = *a.add(i) * *b.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(av, bv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(av, bv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                }
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(av, bv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_mul_ps(av, bv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) =
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_muladd_assign(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            let av = _mm512_loadu_ps(a.add(i));
            let bv = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_fmadd_ps(av, bv, yv));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) += *a.add(i) * *b.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) +=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn elem_mulsub_assign(y: *mut f32, a: *const f32, b: *const f32, n: usize) {
    let vec_end = n & !15;
    let mut i = 0;
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            let av = _mm512_loadu_ps(a.add(i));
            let bv = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_fnmadd_ps(av, bv, yv));
        }
        i += 16;
    }
    while i < n {
        unsafe { *y.add(i) -= *a.add(i) * *b.add(i) };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fnmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(a.add(i * lda as usize + j));
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fnmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(i * lda as usize + j) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(b.add(i * ldb as usize + j));
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fnmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(i * ldb as usize + j)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    let vec_end = n & !15;
    for i in 0..m {
        let mut j = 0;
        while j < vec_end {
            unsafe {
                let mut tmp_a = [0.0; 16];
                let mut tmp_b = [0.0; 16];
                for k in 0..16 {
                    tmp_a[k] = *a.add((j + k) * lda as usize + i);
                    tmp_b[k] = *b.add((j + k) * ldb as usize + i);
                }
                let yv = _mm512_loadu_ps(y.add(i * ldy as usize + j));
                let av = _mm512_loadu_ps(tmp_a.as_ptr());
                let bv = _mm512_loadu_ps(tmp_b.as_ptr());
                _mm512_storeu_ps(y.add(i * ldy as usize + j), _mm512_fnmadd_ps(av, bv, yv));
            }
            j += 16;
        }
        while j < n {
            unsafe {
                *y.add(i * ldy as usize + j) -=
                    *a.add(j * lda as usize + i) * *b.add(j * ldb as usize + i)
            };
            j += 1;
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn rms(x: *const f32, n: usize) -> f32 {
    let vec_end = unsafe { x.add(n & !15) };
    let end = unsafe { x.add(n) };
    let mut ptr = x;
    let mut acc = unsafe { _mm512_setzero_ps() };

    while ptr != vec_end {
        unsafe {
            let xv = _mm512_loadu_ps(ptr);
            let sq = _mm512_mul_ps(xv, xv);
            acc = _mm512_add_ps(acc, sq);
            ptr = ptr.add(16);
        }
    }

    let mut sq_sum = unsafe { _mm512_reduce_add_ps(acc) };
    while ptr != end {
        unsafe {
            let v = *ptr;
            sq_sum += v * v;
            ptr = ptr.add(1);
        }
    }

    (sq_sum / (n as f32)).sqrt()
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn rope_cos(y: *mut f32, k: f32, theta: f32, d: f32, n: usize) {
    let vec_end = unsafe { y.add(n & !15) };
    let end = unsafe { y.add(n) };
    let mut ptr = y;
    let mut i: usize = 0;

    while ptr != vec_end {
        unsafe {
            let mut tmp: [f32; 16] = [0.0; 16];
            for j in 0..16 {
                let idx = i + j;
                let angle = k / theta.powf(2.0 * (idx as f32) / d);
                tmp[j] = angle.cos();
            }

            let out_v = _mm512_loadu_ps(tmp.as_ptr());
            _mm512_storeu_ps(ptr, out_v);

            ptr = ptr.add(16);
        }
        i += 16;
    }

    while ptr != end {
        unsafe {
            let angle = k / theta.powf(2.0 * (i as f32) / d);
            *ptr = angle.cos();
            ptr = ptr.add(1);
        }
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn rope_sin(y: *mut f32, k: f32, theta: f32, d: f32, n: usize) {
    let vec_end = unsafe { y.add(n & !15) };
    let end = unsafe { y.add(n) };
    let mut ptr = y;
    let mut i: usize = 0;

    while ptr != vec_end {
        unsafe {
            let mut tmp: [f32; 16] = [0.0; 16];
            for j in 0..16 {
                let idx = i + j;
                let angle = k / theta.powf(2.0 * (idx as f32) / d);
                tmp[j] = angle.sin();
            }

            let out_v = _mm512_loadu_ps(tmp.as_ptr());
            _mm512_storeu_ps(ptr, out_v);

            ptr = ptr.add(16);
        }
        i += 16;
    }

    while ptr != end {
        unsafe {
            let angle = k / theta.powf(2.0 * (i as f32) / d);
            *ptr = angle.sin();
            ptr = ptr.add(1);
        }
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn masked_safe_softmax(y: *mut f32, n_mask: usize, n: usize) {
    let end = n.saturating_sub(n_mask);
    if end == 0 {
        let mut k = 0;
        while k < n {
            unsafe { *y.add(k) = 0.0 };
            k += 1;
        }
        return;
    }

    let mut max_val = f32::NEG_INFINITY;
    let vec_end = end & !15;
    let mut i = 0;

    let mut v_max = unsafe { _mm512_set1_ps(f32::NEG_INFINITY) };
    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            let mask = _mm512_cmp_ps_mask(yv, v_max, _CMP_GT_OQ);
            v_max = _mm512_mask_blend_ps(mask, v_max, yv);
        }
        i += 16;
    }

    let mut tmp_max = [0.0f32; 16];
    unsafe { _mm512_storeu_ps(tmp_max.as_mut_ptr(), v_max) };
    for k in 0..16 {
        if tmp_max[k] > max_val {
            max_val = tmp_max[k];
        }
    }

    while i < end {
        let v = unsafe { *y.add(i) };
        if v > max_val {
            max_val = v;
        }
        i += 1;
    }

    let mut sum = 0.0;
    i = 0;
    while i < end {
        let exp_v = unsafe { (*y.add(i) - max_val).exp() };
        unsafe { *y.add(i) = exp_v };
        sum += exp_v;
    }

    let multiplier = 1.0 / sum;
    let v_mult = unsafe { _mm512_set1_ps(multiplier) };
    i = 0;

    while i < vec_end {
        unsafe {
            let yv = _mm512_loadu_ps(y.add(i));
            _mm512_storeu_ps(y.add(i), _mm512_mul_ps(yv, v_mult));
        }
        i += 16;
    }
    while i < end {
        unsafe { *y.add(i) *= multiplier };
        i += 1;
    }

    while i < n {
        unsafe { *y.add(i) = 0.0 };
        i += 1;
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn silu(y: *mut f32, n: usize) {
    let vec_end = unsafe { y.add(n & !15) };
    let end = unsafe { y.add(n) };
    let mut ptr = y;

    while ptr != vec_end {
        unsafe {
            let mut tmp: [f32; 16] = [0.0; 16];
            let v = _mm512_loadu_ps(ptr);
            _mm512_storeu_ps(tmp.as_mut_ptr(), v);
            for i in 0..16 {
                let vi = tmp[i];
                tmp[i] = vi / (1.0 + (-vi).exp());
            }

            let out_v = _mm512_loadu_ps(tmp.as_ptr());
            _mm512_storeu_ps(ptr, out_v);
            ptr = ptr.add(16);
        }
    }

    while ptr != end {
        unsafe {
            let v = *ptr;
            *ptr = v / (1.0 + (-v).exp());
            ptr = ptr.add(1);
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bk in (0..k).step_by(BLOCK_SIZE) {
            for bj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let k_end = (bk + BLOCK_SIZE).min(k);
                let j_end = (bj + BLOCK_SIZE).min(n);

                for i in bi..i_end {
                    for k_idx in bk..k_end {
                        unsafe {
                            let a_val = *a.add(i * lda as usize + k_idx);
                            let a_vec = _mm512_set1_ps(a_val);
                            let mut j = bj;
                            while j + 15 < j_end {
                                let c_ptr = y.add(i * ldy as usize + j);
                                let b_ptr = b.add(k_idx * ldb as usize + j);
                                let c_vec = _mm512_loadu_ps(c_ptr);
                                let b_vec = _mm512_loadu_ps(b_ptr);
                                let out = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                _mm512_storeu_ps(c_ptr, out);
                                j += 16;
                            }
                            while j < j_end {
                                let b_val = *b.add(k_idx * ldb as usize + j);
                                *y.add(i * ldy as usize + j) += a_val * b_val;
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        unsafe {
                            let mut sum_vec = _mm512_setzero_ps();
                            let mut k_idx = bk;

                            while k_idx + 15 < k_end {
                                let a_vec = _mm512_loadu_ps(a.add(i * lda as usize + k_idx));
                                let b_vec = _mm512_loadu_ps(b.add(k_idx + j * ldb as usize));
                                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                                k_idx += 16;
                            }

                            let mut sum = if bk == 0 {
                                0.0
                            } else {
                                *y.add(i * ldy as usize + j)
                            };
                            sum += _mm512_reduce_add_ps(sum_vec);

                            while k_idx < k_end {
                                sum += (*a.add(i * lda as usize + k_idx))
                                    * (*b.add(k_idx + j * ldb as usize));
                                k_idx += 1;
                            }
                            *y.add(i * ldy as usize + j) = sum;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bk in (0..k).step_by(BLOCK_SIZE) {
            for bj in (0..n).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let k_end = (bk + BLOCK_SIZE).min(k);
                let j_end = (bj + BLOCK_SIZE).min(n);

                for i in bi..i_end {
                    for k_idx in bk..k_end {
                        unsafe {
                            let a_val = *a.add(k_idx * lda as usize + i);
                            let a_vec = _mm512_set1_ps(a_val);
                            let mut j = bj;
                            while j + 15 < j_end {
                                let c_ptr = y.add(i * ldy as usize + j);
                                let b_ptr = b.add(k_idx * ldb as usize + j);
                                let c_vec = _mm512_loadu_ps(c_ptr);
                                let b_vec = _mm512_loadu_ps(b_ptr);
                                let out = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                _mm512_storeu_ps(c_ptr, out);
                                j += 16;
                            }
                            while j < j_end {
                                let b_val = *b.add(k_idx * ldb as usize + j);
                                *y.add(i * ldy as usize + j) += a_val * b_val;
                                j += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
    for bi in (0..m).step_by(BLOCK_SIZE) {
        for bj in (0..n).step_by(BLOCK_SIZE) {
            for bk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (bi + BLOCK_SIZE).min(m);
                let j_end = (bj + BLOCK_SIZE).min(n);
                let k_end = (bk + BLOCK_SIZE).min(k);

                for i in bi..i_end {
                    for j in bj..j_end {
                        unsafe {
                            let mut sum_vec = _mm512_setzero_ps();
                            let mut k_idx = bk;

                            while k_idx + 15 < k_end {
                                let a_vec = _mm512_loadu_ps(a.add(k_idx * lda as usize + i));
                                let b_vec = _mm512_loadu_ps(b.add(k_idx + j * ldb as usize));
                                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                                k_idx += 16;
                            }

                            let mut sum = if bk == 0 {
                                0.0
                            } else {
                                *y.add(i * ldy as usize + j)
                            };
                            sum += _mm512_reduce_add_ps(sum_vec);

                            while k_idx < k_end {
                                sum += (*a.add(k_idx * lda as usize + i))
                                    * (*b.add(k_idx + j * ldb as usize));
                                k_idx += 1;
                            }
                            *y.add(i * ldy as usize + j) = sum;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
            unsafe {
                temp[j] = *y.add(i * ldy as usize + j);
                *y.add(i * ldy as usize + j) = 0.0;
            }
        }
        for k in 0..n {
            unsafe {
                let t_val = temp[k];
                let t_vec = _mm512_set1_ps(t_val);
                let mut j = 0;
                while j + 15 < n {
                    let y_ptr = y.add(i * ldy as usize + j);
                    let a_ptr = a.add(k * lda as usize + j);
                    let y_vec = _mm512_loadu_ps(y_ptr);
                    let a_vec = _mm512_loadu_ps(a_ptr);
                    _mm512_storeu_ps(y_ptr, _mm512_fmadd_ps(t_vec, a_vec, y_vec));
                    j += 16;
                }
                while j < n {
                    *y.add(i * ldy as usize + j) += t_val * *a.add(k * lda as usize + j);
                    j += 1;
                }
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
            unsafe { temp[j] = *y.add(i * ldy as usize + j) };
        }
        for j in 0..n {
            unsafe {
                let mut sum_vec = _mm512_setzero_ps();
                let mut k = 0;
                while k + 15 < n {
                    let t_vec = _mm512_loadu_ps(temp.as_ptr().add(k));
                    let a_vec = _mm512_loadu_ps(a.add(j * lda as usize + k));
                    sum_vec = _mm512_fmadd_ps(t_vec, a_vec, sum_vec);
                    k += 16;
                }
                let mut sum = _mm512_reduce_add_ps(sum_vec);
                while k < n {
                    sum += temp[k] * *a.add(j * lda as usize + k);
                    k += 1;
                }
                *y.add(i * ldy as usize + j) = sum;
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub unsafe fn matmul_assign_cmn_rnn(
    y: *mut f32,
    ldy: u32,
    a: *const f32,
    lda: u32,
    m: usize,
    n: usize,
) {
    let mut temp = std::vec![0.0; n];
    let mut res = std::vec![0.0; n];
    for i in 0..m {
        for j in 0..n {
            unsafe {
                temp[j] = *y.add(j * ldy as usize + i);
                res[j] = 0.0;
            }
        }
        for k in 0..n {
            unsafe {
                let t_val = temp[k];
                let t_vec = _mm512_set1_ps(t_val);
                let mut j = 0;
                while j + 15 < n {
                    let a_vec = _mm512_loadu_ps(a.add(k * lda as usize + j));
                    let r_vec = _mm512_loadu_ps(res.as_ptr().add(j));
                    _mm512_storeu_ps(
                        res.as_mut_ptr().add(j),
                        _mm512_fmadd_ps(t_vec, a_vec, r_vec),
                    );
                    j += 16;
                }
                while j < n {
                    res[j] += t_val * *a.add(k * lda as usize + j);
                    j += 1;
                }
            }
        }
        for j in 0..n {
            unsafe { *y.add(j * ldy as usize + i) = res[j] };
        }
    }
}

#[cfg(target_feature = "avx512f")]
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
            unsafe { temp[j] = *y.add(j * ldy as usize + i) };
        }
        for j in 0..n {
            unsafe {
                let mut sum_vec = _mm512_setzero_ps();
                let mut k = 0;
                while k + 15 < n {
                    let t_vec = _mm512_loadu_ps(temp.as_ptr().add(k));
                    let a_vec = _mm512_loadu_ps(a.add(j * lda as usize + k));
                    sum_vec = _mm512_fmadd_ps(t_vec, a_vec, sum_vec);
                    k += 16;
                }
                let mut sum = _mm512_reduce_add_ps(sum_vec);
                while k < n {
                    sum += temp[k] * *a.add(j * lda as usize + k);
                    k += 1;
                }
                *y.add(j * ldy as usize + i) = sum;
            }
        }
    }
}
