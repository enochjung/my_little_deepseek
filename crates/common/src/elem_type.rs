pub trait ElemType: Clone + Copy {
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

#[derive(Clone, Copy)]
pub struct BF16(u16);

impl ElemType for BF16 {
    fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        Self(bf16_bits)
    }
    fn to_f32(self) -> f32 {
        let bf16_bits = self.0 as u16;
        let f32_bits = (bf16_bits as u32) << 16;
        f32::from_bits(f32_bits)
    }
}
