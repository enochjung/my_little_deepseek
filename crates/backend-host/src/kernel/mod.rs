#![allow(unused)]

mod avx512_bf16;
mod avx512_f32;
mod unknown_bf16;
mod unknown_f32;

use core::ElemType;

use std::arch::x86_64::bf16;
use std::marker::PhantomData;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::kernel::avx512_f32 as KernelF32;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
use crate::kernel::unknown_f32 as KernelF32;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use crate::kernel::avx512_bf16 as KernelBF16;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
use crate::kernel::unknown_bf16 as KernelBF16;

pub struct Kernel<T: ElemType> {
    _phantom: PhantomData<T>,
}

macro_rules! define_kernel_ops {
    (
        $(
            unsafe fn $name:ident ( $( $arg:ident : $arg_ty:ty ),* ) $( -> $ret:ty )? ;
        )*
    ) => {
        pub trait KernelOps<T: ElemType> {
            $( unsafe fn $name( $( $arg: $arg_ty ),* ) $( -> $ret )?; )*
        }

        impl<T: ElemType> KernelOps<T> for Kernel<T> {
            $(
                default unsafe fn $name( $($arg: $arg_ty ),* ) $( -> $ret )? {
                    panic!("not implemented")
                }
            )*
        }

        const _: () = {
            type T = f32;
            impl KernelOps<T> for Kernel<T> {
                $(
                    unsafe fn $name( $( $arg: $arg_ty ),* ) $( -> $ret )? {
                        unsafe { KernelF32::$name( $( $arg ),* ) }
                    }
                )*
            }
        };

        const _: () = {
            type T = bf16;
            impl KernelOps<T> for Kernel<T> {
                $(
                    unsafe fn $name( $( $arg: $arg_ty ),* ) $( -> $ret )? {
                        unsafe { KernelBF16::$name( $( $arg ),* ) }
                    }
                )*
            }
        };
    };
}

define_kernel_ops! {
    unsafe fn elem_add_assign(y: *mut T, x: *const T, n: usize);
    unsafe fn elem_add_assign_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn elem_add_assign_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn elem_add(y: *mut T, a: *const T, b: *const T, n: usize);
    unsafe fn elem_add_rmn_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_add_rmn_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_add_rmn_cmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_add_rmn_cmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn argmax(x: *const T, n: usize) -> u32;
    unsafe fn copy(y: *mut T, x: *const T, n: usize);
    unsafe fn copy_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn copy_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn fill(y: *mut T, value: T, n: usize);
    unsafe fn scalar_mul_assign(y: *mut T, value: T, n: usize);
    unsafe fn elem_mul_assign(y: *mut T, x: *const T, n: usize);
    unsafe fn elem_mul_assign_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn elem_mul_assign_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn elem_mul(y: *mut T, a: *const T, b: *const T, n: usize);
    unsafe fn elem_mul_rmn_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mul_rmn_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mul_rmn_cmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mul_rmn_cmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_muladd_assign(y: *mut T, a: *const T, b: *const T, n: usize);
    unsafe fn elem_muladd_assign_rmn_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_muladd_assign_rmn_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_muladd_assign_rmn_cmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_muladd_assign_rmn_cmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mulsub_assign(y: *mut T, a: *const T, b: *const T, n: usize);
    unsafe fn elem_mulsub_assign_rmn_rmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mulsub_assign_rmn_rmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mulsub_assign_rmn_cmn_rmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn elem_mulsub_assign_rmn_cmn_cmn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, n: usize);
    unsafe fn rms(x: *const T, n: usize) -> T;
    unsafe fn rope_cos(y: *mut T, k: T, theta: T, d: T, n: usize);
    unsafe fn rope_sin(y: *mut T, k: T, theta: T, d: T, n: usize);
    unsafe fn safe_softmax(y: *mut T, n: usize);
    unsafe fn silu(y: *mut T, n: usize);
    unsafe fn matmul_rmn_rmk_rkn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, k: usize, n: usize);
    unsafe fn matmul_rmn_rmk_ckn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, k: usize, n: usize);
    unsafe fn matmul_rmn_cmk_rkn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, k: usize, n: usize);
    unsafe fn matmul_rmn_cmk_ckn(y: *mut T, ldy: u32, a: *const T, lda: u32, b: *const T, ldb: u32, m: usize, k: usize, n: usize);
    unsafe fn matmul_assign_rmn_rnn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn matmul_assign_rmn_cnn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn matmul_assign_cmn_rnn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
    unsafe fn matmul_assign_cmn_cnn(y: *mut T, ldy: u32, a: *const T, lda: u32, m: usize, n: usize);
}
