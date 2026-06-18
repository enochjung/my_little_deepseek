use crate::host::Host;
use crate::mmap::Mmap;
use core::{Backend, MatrixLayout, Memory, MemoryMut};

impl Backend<f16> for Host<f16> {
    type Memory = Mmap<f16>;

    unsafe fn elem_add_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src: &S0,
        src_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn elem_br_add_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src: &S0,
        src_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn argmax<S0: Memory<f16, Base = Self::Memory>>(
        src: &S0,
        src_ml: &MatrixLayout<f16>,
    ) -> u32 {
        todo!()
    }

    unsafe fn copy<D: MemoryMut<f16, Base = Self::Memory>, S0: Memory<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src: &S0,
        src_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn fill<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        value: f16,
    ) {
        todo!()
    }

    unsafe fn scalar_mul_assign<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        value: f16,
    ) {
        todo!()
    }

    unsafe fn elem_mul<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
        S1: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src0: &S0,
        src0_ml: &MatrixLayout<f16>,
        src1: &S1,
        src1_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn elem_mul_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src: &S0,
        src_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn elem_muladd_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
        S1: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src0: &S0,
        src0_ml: &MatrixLayout<f16>,
        src1: &S1,
        src1_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn elem_mulsub_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
        S1: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src0: &S0,
        src0_ml: &MatrixLayout<f16>,
        src1: &S1,
        src1_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn matmul<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
        S1: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src0: &S0,
        src0_ml: &MatrixLayout<f16>,
        src1: &S1,
        src1_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn matmul_assign<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &MatrixLayout<f16>,
        _src: &S0,
        _src_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn rms_norm<
        D: MemoryMut<f16, Base = Self::Memory>,
        S0: Memory<f16, Base = Self::Memory>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        src: &S0,
        src_ml: &MatrixLayout<f16>,
        epsilon: f16,
    ) {
        todo!()
    }

    unsafe fn rope_cos<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        k: f16,
        theta: f16,
        d: f16,
    ) {
        todo!()
    }

    unsafe fn rope_sin<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
        k: f16,
        theta: f16,
        d: f16,
    ) {
        todo!()
    }

    unsafe fn safe_softmax<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }

    unsafe fn silu<D: MemoryMut<f16, Base = Self::Memory>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<f16>,
    ) {
        todo!()
    }
}
