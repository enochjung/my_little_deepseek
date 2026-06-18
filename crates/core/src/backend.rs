use crate::elem_type::ElemType;
use crate::matrix_layout::MatrixLayout;
use crate::memory::{Memory, MemoryMut, MemoryOwn};

pub trait Backend<T> {
    type Operand: MemoryOwn<T, Operator = Self>;
}

pub trait BackendOps<T: ElemType>: Backend<T> {
    /// Adds `src` to `dst` element-wise.
    ///
    /// Conceptually: `dst[i, j] += src[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst` and `src` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst` and `src` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn elem_add_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
    );

    /// Adds the first row of `src` to all rows of `dst` (broadcasting).
    ///
    /// Conceptually: `dst[i, j] += src[0, j]`
    /// Physical index:
    /// - `dst_idx = dst_ml.offset + (i * dst_ml.row_stride) + (j * dst_ml.col_stride)`
    /// - `src_idx = src_ml.offset + (j * src_ml.col_stride)`
    ///
    /// # Safety
    /// - `dst` and `src` must have an identical number of columns (`ncol`).
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Pointers derived from `src` must be valid for reads up to `offset + (ncol - 1) * col_stride`.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn elem_br_add_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
    );

    /// Finds the index of the maximum value in the first row of `src`.
    ///
    /// Conceptually: Returns `i` such that `src[0, i] == max(src[0, 0..ncol])`.
    /// Ties are broken arbitrarily.
    /// Physical index: `idx = offset + (i * col_stride)`
    ///
    /// # Safety
    /// - `src` must have exactly 1 row (`nrow == 1`).
    /// - `src` must have a column stride of 1 (`col_stride == 1`).
    /// - Pointers derived from `src` must be valid for reads up to `offset + (ncol - 1) * col_stride`.
    unsafe fn argmax<S0: Memory<T, Base = Self::Operand>>(
        src: &S0,
        src_ml: &MatrixLayout<T>,
    ) -> u32;

    /// Copies values from `src` to `dst`.
    ///
    /// Conceptually: `dst[i, j] = src[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst` and `src` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst` and `src` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn copy<D: MemoryMut<T, Base = Self::Operand>, S0: Memory<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
    );

    /// Fills all elements of `dst` with a given scalar value.
    ///
    /// Conceptually: `dst[i, j] = value`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - Pointers derived from `dst` must be valid for writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn fill<D: MemoryMut<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        value: T,
    );

    /// Multiplies all elements of `dst` by a given scalar value in-place.
    ///
    /// Conceptually: `dst[i, j] *= value`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn scalar_mul_assign<D: MemoryMut<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        value: T,
    );

    /// Computes the element-wise product of `src0` and `src1` into `dst`.
    ///
    /// Conceptually: `dst[i, j] = src0[i, j] * src1[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst`, `src0`, and `src1` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst`, `src0`, and `src1` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory region of `dst` must not overlap with `src0` or `src1`.
    unsafe fn elem_mul<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
        S1: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
    );

    /// Multiplies `dst` by `src` element-wise in-place.
    ///
    /// Conceptually: `dst[i, j] *= src[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst` and `src` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst` and `src` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn elem_mul_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
    );

    /// Computes the element-wise product of `src0` and `src1` and adds it to `dst`.
    ///
    /// Conceptually: `dst[i, j] += src0[i, j] * src1[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst`, `src0`, and `src1` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst`, `src0`, and `src1` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory region of `dst` must not overlap with `src0` or `src1`.
    unsafe fn elem_muladd_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
        S1: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
    );

    /// Computes the element-wise product of `src0` and `src1` and subtracts it from `dst`.
    ///
    /// Conceptually: `dst[i, j] -= src0[i, j] * src1[i, j]`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst`, `src0`, and `src1` must have identical dimensions (`nrow` and `ncol`).
    /// - Pointers derived from `dst`, `src0`, and `src1` must be valid for reads/writes up to their respective maximum physical indices:
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Memory region of `dst` must not overlap with `src0` or `src1`.
    unsafe fn elem_mulsub_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
        S1: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
    );

    /// Performs matrix multiplication of `src0` and `src1` into `dst`.
    ///
    /// Conceptually: `dst = src0 @ src1`
    /// Physical index:
    /// - `dst[i, j]: dst_ml.offset + (i * dst_ml.row_stride) + (j * dst_ml.col_stride)`
    /// - `src0[i, k]: src0_ml.offset + (i * src0_ml.row_stride) + (k * src0_ml.col_stride)`
    /// - `src1[k, j]: src1_ml.offset + (k * src1_ml.row_stride) + (j * src1_ml.col_stride)`
    ///
    /// # Safety
    /// - Matrix dimensions must be compatible for multiplication (`src0_ml.ncol == src1_ml.nrow`, `dst_ml.nrow == src0_ml.nrow`, `dst_ml.ncol == src1_ml.ncol`).
    /// - Pointers derived from `dst`, `src0`, and `src1` must be valid for reads/writes up to their respective maximum physical indices.
    /// - Memory region of `dst` must not overlap with `src0` or `src1`.
    unsafe fn matmul<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
        S1: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src0: &S0,
        src0_ml: &MatrixLayout<T>,
        src1: &S1,
        src1_ml: &MatrixLayout<T>,
    );

    /// Performs in-place matrix multiplication of `dst` and `src`.
    ///
    /// Conceptually: `dst = dst @ src`
    /// Physical index:
    /// - `dst[i, k]: dst_ml.offset + (i * dst_ml.row_stride) + (k * dst_ml.col_stride)`
    /// - `src[k, j]: src_ml.offset + (k * src_ml.row_stride) + (j * src_ml.col_stride)`
    ///
    /// # Safety
    /// - Matrix dimensions must be compatible and support in-place update (`dst_ml.ncol == src_ml.nrow` and `src_ml.nrow == src_ml.ncol`).
    /// - Pointers derived from `dst` and `src` must be valid for reads/writes up to their respective maximum physical indices.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn matmul_assign<
        D: MemoryMut<T, Base = Self::Operand>,
        S0: Memory<T, Base = Self::Operand>,
    >(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
    );

    /// Applies Root Mean Square (RMS) normalization to `dst` and scales by `src`.
    ///
    /// Conceptually: `dst[i, j] = dst[i, j] * src[0, j] / (rms[i] + epsilon)`
    /// where `rms[i] = sqrt(sum_k(dst[i, k]^2) / ncol)`
    /// Physical index:
    /// - `dst[i, j]: dst_ml.offset + (i * dst_ml.row_stride) + (j * dst_ml.col_stride)`
    /// - `src[0, j]: src_ml.offset + (j * src_ml.col_stride)`
    ///
    /// # Safety
    /// - `dst` and `src` must have an identical number of columns (`ncol`).
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    /// - Pointers derived from `src` must be valid for reads up to `offset + (ncol - 1) * col_stride`.
    /// - Memory regions of `dst` and `src` must not overlap.
    unsafe fn rms_norm<D: MemoryMut<T, Base = Self::Operand>, S0: Memory<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        src: &S0,
        src_ml: &MatrixLayout<T>,
        epsilon: f32,
    );

    /// Applies Rotary Position Embedding (RoPE) cosine frequencies to `dst`.
    ///
    /// Conceptually: `dst[i] = cos(k / theta^(2i / d))` iteratively for each element.
    /// Physical index: `idx = offset + (i * col_stride)`
    ///
    /// # Safety
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn rope_cos<D: MemoryMut<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        k: T,
        theta: T,
        d: T,
    );

    /// Applies Rotary Position Embedding (RoPE) sine frequencies to `dst`.
    ///
    /// Conceptually: `dst[i] = sin(k / theta^(2i / d))` iteratively for each element.
    /// Physical index: `idx = offset + (i * col_stride)`
    ///
    /// # Safety
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn rope_sin<D: MemoryMut<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        k: T,
        theta: T,
        d: T,
    );

    /// Applies a masked, numerically safe softmax function to `dst` in-place.
    ///
    /// Conceptually, for the first row (`i = 0`):
    /// - Performs safe softmax on `dst[0, 0 .. ncol - n_mask]`.
    /// - Fills the masked region `dst[0, ncol - n_mask .. ncol]` with zeros.
    ///   (Softmax: `x_j = exp(x_j - max) / sum(exp(x_k - max))`)
    ///
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - `dst` must be row-major (`col_stride == 1`).
    /// - Pointers derived from `dst` must be valid for reads/writes up to
    ///   `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn masked_safe_softmax<D: MemoryMut<T, Base = Self::Operand>>(
        dst: &mut D,
        dst_ml: &MatrixLayout<T>,
        n_mask: u32,
    );

    /// Applies the SiLU (Swish) activation function to `dst` in-place.
    ///
    /// Conceptually: `dst[i, j] = dst[i, j] / (1 + exp(-dst[i, j]))`
    /// Physical index: `idx = offset + (i * row_stride) + (j * col_stride)`
    ///
    /// # Safety
    /// - Pointers derived from `dst` must be valid for reads/writes up to `offset + (nrow - 1) * row_stride + (ncol - 1) * col_stride`.
    unsafe fn silu<D: MemoryMut<T, Base = Self::Operand>>(dst: &mut D, dst_ml: &MatrixLayout<T>);
}
