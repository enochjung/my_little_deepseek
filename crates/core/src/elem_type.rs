use std::arch::x86_64::bf16;

pub trait CastUnalignedBF16 {
    unsafe fn cast_bf16(dst: *mut Self, src: *const u8, count: usize);
}

pub trait ElemType: Clone + Copy + Send + Sync + CastUnalignedBF16 {
    fn from_f32(value: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl ElemType for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl CastUnalignedBF16 for f32 {
    unsafe fn cast_bf16(dst: *mut Self, src: *const u8, count: usize) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16"))]
        return unsafe { cast_f32_to_bf16_avx512(dst, src, count) };

        let dst_end = unsafe { dst.add(count) };
        let mut dst = dst;
        let mut src = src;

        while dst != dst_end {
            let bf16_u16 = unsafe { (src as *const u16).read_unaligned() } as u32;
            unsafe { *dst = f32::from_bits(bf16_u16 << 16) };
            dst = unsafe { dst.add(1) };
            src = unsafe { src.byte_add(size_of::<bf16>()) };
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16"))]
unsafe fn cast_f32_to_bf16_avx512(dst: *mut f32, src: *const u8, count: usize) {
    let vec_end = unsafe { dst.add(count & !15) };
    let dst_end = unsafe { dst.add(count) };
    let mut dst = dst;
    let mut src = src;

    while dst != vec_end {
        let bf16_v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let bits_u32 = _mm512_cvtepu16_epi32(bf16_v);
        let bits_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(bits_u32, 16));
        unsafe { _mm512_store_ps(dst, bits_f32) };
        dst = unsafe { dst.add(16) };
        src = unsafe { src.byte_add(16 * size_of::<bf16>()) };
    }

    while dst != dst_end {
        let bf16_u16 = unsafe { (src as *const u16).read_unaligned() } as u32;
        unsafe { *dst = f32::from_bits(bf16_u16 << 16) };
        dst = unsafe { dst.add(1) };
        src = unsafe { src.byte_add(size_of::<f16>()) };
    }

    /////////
    ///
    ///
    use std::arch::x86_64::*;
    use std::mem::size_of;

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vl",
        target_feature = "avx512fp16"
    ))]
    unsafe fn cast_f16_to_f32_avx512(dst: *mut f32, src: *const u8, count: usize) {
        // 16개씩 처리 (512비트 레지스터 기준)
        let vec_end = dst.add(count & !15);
        let dst_end = dst.add(count);
        let mut dst = dst;
        let mut src = src;

        // Vectorized path: 16개씩 처리
        while dst < vec_end {
            // 16-bit f16 16개를 로드 (256비트 레지스터 사용)
            let f16_v = _mm256_loadu_si256(src as *const __m256i);

            // f16 -> f32 변환 (AVX-512 FP16 명령어)
            let f32_v = _mm512_cvtph_ps(f16_v);

            // 결과 저장
            _mm512_storeu_ps(dst, f32_v);

            dst = dst.add(16);
            src = src.add(16 * 2); // f16은 2바이트
        }

        // Scalar path: 나머지 처리
        while dst < dst_end {
            let f16_bits = (src as *const u16).read_unaligned();
            // f16을 f32로 변환 (crate 사용 권장, 여기선 로직만 표시)
            *dst = f32::from(half::f16::from_bits(f16_bits));
            dst = dst.add(1);
            src = src.add(2);
        }
    }
}

impl ElemType for bf16 {
    fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        bf16::from_bits(bf16_bits)
    }

    fn to_f32(self) -> f32 {
        let bf16_bits = self.to_bits() as u32;
        let f32_bits = bf16_bits << 16;
        f32::from_bits(f32_bits)
    }
}

impl CastUnalignedBF16 for bf16 {
    unsafe fn cast_bf16(dst: *mut Self, src: *const u8, count: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst as *mut u8, count * size_of::<bf16>()) }
    }
}
