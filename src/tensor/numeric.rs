pub(crate) trait Numeric {
    const BYTES: usize;
}

pub(crate) struct BF16;
impl Numeric for BF16 {
    const BYTES: usize = 2;
}

pub(crate) struct F32;
impl Numeric for F32 {
    const BYTES: usize = 4;
}
