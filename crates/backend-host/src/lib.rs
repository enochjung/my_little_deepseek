mod host;
mod kernel;
mod mem;

pub use host::Host;

/*
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
*/

/*
impl CastUnalignedBF16 for bf16 {
    unsafe fn cast_bf16(dst: *mut Self, src: *const u8, count: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst as *mut u8, count * size_of::<bf16>()) }
    }
}
    */
