use core::ElemType;
use std::marker::PhantomData;

pub struct Host<T: ElemType> {
    _phantom: PhantomData<T>,
}

/*
impl<T: ElemType> Backend<T> for Host<T> {
    type Memory = Mmap<T>;

    unsafe fn elem_add_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn elem_br_add_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn argmax<S0: Memory<T, Base = Self::Memory>>(
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) -> u32 {
        todo!()
    }

    unsafe fn copy<D: MemoryMut<T, Base = Self::Memory>, S0: Memory<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn fill<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _value: T,
    ) {
        todo!()
    }

    unsafe fn scalar_mul_assign<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _value: T,
    ) {
        todo!()
    }

    unsafe fn elem_mul<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src0: &S0,
        _src0_ml: &core::MatrixLayout<T>,
        _src1: &S1,
        _src1_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn elem_mul_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn elem_muladd_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src0: &S0,
        _src0_ml: &core::MatrixLayout<T>,
        _src1: &S1,
        _src1_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn elem_mulsub_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src0: &S0,
        _src0_ml: &core::MatrixLayout<T>,
        _src1: &S1,
        _src1_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn matmul<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
        S1: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src0: &S0,
        _src0_ml: &core::MatrixLayout<T>,
        _src1: &S1,
        _src1_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn matmul_assign<
        D: MemoryMut<T, Base = Self::Memory>,
        S0: Memory<T, Base = Self::Memory>,
    >(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn rms_norm<D: MemoryMut<T, Base = Self::Memory>, S0: Memory<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _src: &S0,
        _src_ml: &core::MatrixLayout<T>,
        _epsilon: T,
    ) {
        todo!()
    }

    unsafe fn rope_cos<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _k: T,
        _theta: T,
        _d: T,
    ) {
        todo!()
    }

    unsafe fn rope_sin<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
        _k: T,
        _theta: T,
        _d: T,
    ) {
        todo!()
    }

    unsafe fn safe_softmax<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }

    unsafe fn silu<D: MemoryMut<T, Base = Self::Memory>>(
        _dst: &mut D,
        _dst_ml: &core::MatrixLayout<T>,
    ) {
        todo!()
    }
}
    */

/*
impl DeviceOps<F32> for Mmap {
    unsafe fn add<M: MutableDevice<Base = Self::Memory>, D0: Device<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src: &D0,
        _src_ml: &MatrixLayout,
    )  {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_ml.offset) } as *const f32;
        if dst_ml.is_packed() && src_ml.is_packed() {
            let n = (dst_ml.nrow * dst_ml.ncol) as usize;
            unsafe { kernel::add_n_n(dst, src, n) };
        } else {
            let n = dst_ml.ncol as usize;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_ml.nrow {
                unsafe { kernel::add_n_n(dst, src, n) };
                _dst = unsafe { dst.add(dst_ml.stride as usize) };
                _src = unsafe { src.add(src_ml.stride as usize) };
            }
        }
    }
    unsafe fn argmax<D: Device<Base = Self::Memory>>(src: &D, src_ml: &MatrixLayout) -> u32 {
        let src = unsafe { src.as_ptr().byte_add(src_ml.offset) } as *mut f32;
        let n = src_ml.ncol as usize;
        unsafe { kernel::argmax_n(src, n) }
    }
    unsafe fn cast_from_bf16<M: MutableDevice<Base = Self::Memory>, D0: Device<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src: &D0,
        _src_ml: &MatrixLayout,
    )  {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        if dst_ml.is_packed() && src_ml.is_packed() {
            let n = (dst_ml.nrow * dst_ml.ncol) as usize;
            unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
        } else {
            let n = dst_ml.ncol as usize;
            let src_stride_bytes = src_ml.stride as usize * BF16::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_ml.nrow {
                unsafe { kernel::cast_bf16_to_f32_n_n(dst, src, n) };
                _dst = unsafe { dst.add(dst_ml.stride as usize) };
                _src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }
    }
    unsafe fn copy<M: MutableDevice<Base = Self::Memory>, D0: Device<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src: &D0,
        _src_ml: &MatrixLayout,
    )  {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) };
        let src = unsafe { src.as_ptr().byte_add(src_ml.offset) };

        if dst_ml.is_packed() && src_ml.is_packed() {
            let len = (dst_ml.nrow * dst_ml.ncol) as usize * F32::BYTES;
            unsafe { kernel::copy(dst, src, len) };
        } else {
            let len = dst_ml.ncol as usize * F32::BYTES;
            let dst_stride_bytes = dst_ml.stride as usize * F32::BYTES;
            let src_stride_bytes = src_ml.stride as usize * F32::BYTES;

            let mut dst = dst;
            let mut src = src;
            for _ in 0..dst_ml.nrow {
                unsafe { kernel::copy(dst, src, len) };
                _dst = unsafe { dst.byte_add(dst_stride_bytes) };
                _src = unsafe { src.byte_add(src_stride_bytes) };
            }
        }
    }
    unsafe fn mul_elementwise<
        M: MutableDevice<Base = Self::Memory>,
        D0: Device<Base = Self::Memory>,
        D1: Device<Base = Self::Memory>,
    >(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src0: &D0,
        _src0_ml: &MatrixLayout,
        _src1: &D1,
        _src1_ml: &MatrixLayout,
        alpha: f32,
    )  {
        unsafe { Self::copy(dst, dst_ml, src0, src0_ml) };

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let src1 = unsafe { src1.as_ptr().byte_add(src1_ml.offset) } as *const f32;

        if dst_ml.is_packed() && src1_ml.is_packed() {
            let len = (dst_ml.nrow * dst_ml.ncol) as usize;
            unsafe { kernel::mul_n_n(dst, src1, alpha, len) };
        } else {
            let len = dst_ml.ncol as usize;

            let mut dst = dst;
            let mut src1 = src1;
            for _ in 0..dst_ml.nrow {
                unsafe { kernel::mul_n_n(dst, src1, alpha, len) };
                _dst = unsafe { dst.add(dst_ml.stride as usize) };
                _src1 = unsafe { src1.add(src1_ml.stride as usize) };
            }
        }
    }
    unsafe fn mul_mn_mk_kn<
        M: MutableDevice<Base = Self::Memory>,
        D0: Device<Base = Self::Memory>,
        D1: Device<Base = Self::Memory>,
    >(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src0: &D0,
        _src0_ml: &MatrixLayout,
        _src1: &D1,
        _src1_ml: &MatrixLayout,
    )  {
        let m = src0_ml.nrow as usize;
        let k = src0_ml.ncol as usize;
        let n = src1_ml.ncol as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_ml.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_ml.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_rkn(
                _dst,
                _dst_ml.stride,
                a,
                _src0_ml.stride,
                b,
                _src1_ml.stride,
                m,
                k,
                n,
            )
        };
    }
    unsafe fn mul_mn_mk_knt<
        M: MutableDevice<Base = Self::Memory>,
        D0: Device<Base = Self::Memory>,
        D1: Device<Base = Self::Memory>,
    >(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src0: &D0,
        _src0_ml: &MatrixLayout,
        _src1: &D1,
        _src1_ml_t: &MatrixLayout,
    )  {
        let m = src0_ml.nrow as usize;
        let k = src0_ml.ncol as usize;
        let n = src1_ml_t.nrow as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_ml.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_ml_t.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_ckn(
                _dst,
                _dst_ml.stride,
                a,
                _src0_ml.stride,
                b,
                _src1_ml_t.stride,
                m,
                k,
                n,
            )
        };
    }
    unsafe fn mul_mn_mk_knt_1n<
        M: MutableDevice<Base = Self::Memory>,
        D0: Device<Base = Self::Memory>,
        D1: Device<Base = Self::Memory>,
        D2: Device<Base = Self::Memory>,
    >(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src0: &D0,
        _src0_ml: &MatrixLayout,
        _src1: &D1,
        _src1_ml: &MatrixLayout,
        _src2: &D2,
        _src2_ml: &MatrixLayout,
    )  {
        let m = src0_ml.nrow as usize;
        let k = src0_ml.ncol as usize;
        let n = src1_ml.nrow as usize;

        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let a = unsafe { src0.as_ptr().byte_add(src0_ml.offset) } as *const f32;
        let b = unsafe { src1.as_ptr().byte_add(src1_ml.offset) } as *const f32;
        let c = unsafe { src2.as_ptr().byte_add(src2_ml.offset) } as *const f32;

        unsafe {
            kernel::mul_rmn_rmk_ckn_r1n(
                _dst,
                _dst_ml.stride,
                a,
                _src0_ml.stride,
                b,
                _src1_ml.stride,
                c,
                m,
                k,
                n,
            )
        };
    }
    unsafe fn rms_norm<M: MutableDevice<Base = Self::Memory>, D0: Device<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        _src: &D0,
        _src_ml: &MatrixLayout,
        epsilon: f32,
    )  {
        let n = dst_ml.ncol as usize;

        let mut dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let src = unsafe { src.as_ptr().byte_add(src_ml.offset) } as *const f32;
        for _ in 0..dst_ml.nrow {
            let rms = unsafe { kernel::rms_n(dst as *const f32, n) };
            let scale = 1.0 / (rms + epsilon);
            unsafe { kernel::mul_n_n(dst, src, scale, n) };
            _dst = unsafe { dst.add(dst_ml.stride as usize) };
        }
    }
    unsafe fn rope_cos<M: MutableDevice<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        k: f32,
        theta: f32,
        d: f32,
    )  {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let n = dst_ml.ncol as usize;
        unsafe { kernel::rope_cos_n(dst, n, k, theta, d) };
    }
    unsafe fn rope_sin<M: MutableDevice<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        k: f32,
        theta: f32,
        d: f32,
    )  {
        let dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        let n = dst_ml.ncol as usize;
        unsafe { kernel::rope_sin_n(dst, n, k, theta, d) };
    }
    unsafe fn safe_softmax_with_masking<M: MutableDevice<Base = Self::Memory>>(
        _dst: &mut M,
        _dst_ml: &MatrixLayout,
        alpha: f32,
    )  {
        let n = dst_ml.ncol as usize;

        let mut dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        for n_mask in (0..dst_ml.nrow as usize).rev() {
            unsafe { kernel::safe_softmax_with_masking_n(dst, alpha, n_mask, n) };
            _dst = unsafe { dst.add(dst_ml.stride as usize) };
        }
    }
    unsafe fn silu<M: MutableDevice<Base = Self::Memory>>(dst: &mut M, dst_ml: &MatrixLayout)  {
        let n = dst_ml.ncol as usize;

        let mut dst = unsafe { dst.as_mut_ptr().byte_add(dst_ml.offset) } as *mut f32;
        for _ in 0..dst_ml.nrow {
            unsafe { kernel::silu_n(dst, n) };
            _dst = unsafe { dst.add(dst_ml.stride as usize) };
        }
    }
}
*/
