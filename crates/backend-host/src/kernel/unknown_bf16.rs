use std::arch::x86_64::bf16;

pub unsafe fn elem_add_assign(_y: *mut bf16, _x: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_add_assign_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_add_assign_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_add(_y: *mut bf16, _a: *const bf16, _b: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_add_rmn_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_add_rmn_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_add_rmn_cmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_add_rmn_cmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn argmax(_x: *const bf16, _n: usize) -> u32 {
    panic!("not support")
}

pub unsafe fn copy(_y: *mut bf16, _x: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn copy_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn copy_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn fill(_y: *mut bf16, _value: bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn scalar_mul_assign(_y: *mut bf16, _value: bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_mul_assign(_y: *mut bf16, _x: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_mul_assign_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mul_assign_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mul(_y: *mut bf16, _a: *const bf16, _b: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_mul_rmn_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mul_rmn_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mul_rmn_cmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mul_rmn_cmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_muladd_assign(_y: *mut bf16, _a: *const bf16, _b: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_muladd_assign_rmn_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_muladd_assign_rmn_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_muladd_assign_rmn_cmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_muladd_assign_rmn_cmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mulsub_assign(_y: *mut bf16, _a: *const bf16, _b: *const bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn elem_mulsub_assign_rmn_rmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mulsub_assign_rmn_rmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mulsub_assign_rmn_cmn_rmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn elem_mulsub_assign_rmn_cmn_cmn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn rms(_x: *const bf16, _n: usize) -> bf16 {
    panic!("not support")
}

pub unsafe fn rope_cos(_y: *mut bf16, _k: bf16, _theta: bf16, _d: bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn rope_sin(_y: *mut bf16, _k: bf16, _theta: bf16, _d: bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn safe_softmax(_y: *mut bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn silu(_y: *mut bf16, _n: usize) {
    panic!("not support")
}

pub unsafe fn matmul_rmn_rmk_rkn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _k: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_rmn_rmk_ckn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _k: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_rmn_cmk_rkn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _k: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_rmn_cmk_ckn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _b: *const bf16,
    _ldb: u32,
    _m: usize,
    _k: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_assign_rmn_rnn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_assign_rmn_cnn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_assign_cmn_rnn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}

pub unsafe fn matmul_assign_cmn_cnn(
    _y: *mut bf16,
    _ldy: u32,
    _a: *const bf16,
    _lda: u32,
    _m: usize,
    _n: usize,
) {
    panic!("not support")
}
