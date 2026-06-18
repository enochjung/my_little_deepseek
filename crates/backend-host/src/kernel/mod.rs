#[cfg(target_feature = "avx512bf16")]
pub mod avx512_bf16;

#[cfg(target_feature = "avx512f")]
pub mod avx512_f32;
#[cfg(not(target_feature = "avx512f"))]
pub mod unknown_f32;
