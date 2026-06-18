use std::arch::x86_64::bf16;

// TODO: remove
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
