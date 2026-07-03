use core::{BackendOps, ElemType, MLTError, MatrixLayout, Memory, MemoryMut, MemoryOwn};

use std::ops::Range;

pub struct Tensor<T, M: Memory<T>> {
    mem: M,
    ml: MatrixLayout<T>,
}

impl<T, M: Memory<T>> Tensor<T, M> {
    pub fn new(mem: M, ml: MatrixLayout<T>) -> Result<Self, MLTError> {
        validate_space::<T>(mem.size(), &ml)?;
        Ok(Self { mem, ml })
    }

    pub fn transpose(mut self) -> Self {
        self.ml = self.ml.transpose();
        self
    }

    pub fn slice(&self, rows: Range<u32>, cols: Range<u32>) -> Tensor<T, &M::Base> {
        let mem = self.mem.as_base();
        let ml = self.ml.sliced(rows, cols);
        Tensor { mem, ml }
    }
}

impl<T, MO: MemoryOwn<T>> Tensor<T, MO> {
    pub fn reshape(
        &mut self,
        nrow: u32,
        ncol: u32,
        row_stride: u32,
        col_stride: u32,
    ) -> Result<(), MLTError> {
        self.ml.nrow = nrow;
        self.ml.ncol = ncol;
        self.ml.row_stride = row_stride;
        self.ml.col_stride = col_stride;

        let required_size = ((nrow - 1) as usize * row_stride as usize
            + (ncol - 1) as usize * col_stride as usize
            + 1)
            * size_of::<T>();
        if self.mem.size() < required_size {
            self.mem.resize(required_size)?;
        }

        Ok(())
    }
}

impl<T, MM: MemoryMut<T>> Tensor<T, MM> {
    pub fn slice_mut(&mut self, rows: Range<u32>, cols: Range<u32>) -> Tensor<T, &mut MM::Base> {
        let mem = self.mem.as_mut_base();
        let ml = self.ml.sliced(rows, cols);
        Tensor { mem, ml }
    }

    pub fn split_row(
        &mut self,
        mid: u32,
    ) -> Result<(Tensor<T, &mut MM::Base>, Tensor<T, &mut MM::Base>), MLTError> {
        if mid > self.ml.nrow {
            return Err(MLTError::out_of_bound(mid as usize, self.ml.nrow as usize));
        }

        let memptr = self.mem.as_mut_base() as *mut _;
        let t0 = Tensor::new(
            unsafe { &mut *memptr },
            MatrixLayout::new(
                self.ml.offset,
                mid,
                self.ml.ncol,
                self.ml.row_stride,
                self.ml.col_stride,
            ),
        )?;
        let t1 = Tensor::new(
            unsafe { &mut *memptr },
            MatrixLayout::new(
                self.ml.rc_offset(mid, 0),
                self.ml.nrow - mid,
                self.ml.ncol,
                self.ml.row_stride,
                self.ml.col_stride,
            ),
        )?;

        Ok((t0, t1))
    }

    pub fn split_col(
        &mut self,
        mid: u32,
    ) -> Result<(Tensor<T, &mut MM::Base>, Tensor<T, &mut MM::Base>), MLTError> {
        if mid > self.ml.ncol {
            return Err(MLTError::out_of_bound(mid as usize, self.ml.ncol as usize));
        }

        let memptr = self.mem.as_mut_base() as *mut _;
        let t0 = Tensor::new(
            unsafe { &mut *memptr },
            MatrixLayout::new(
                self.ml.offset,
                self.ml.nrow,
                mid,
                self.ml.row_stride,
                self.ml.col_stride,
            ),
        )?;
        let t1 = Tensor::new(
            unsafe { &mut *memptr },
            MatrixLayout::new(
                self.ml.rc_offset(0, mid),
                self.ml.nrow,
                self.ml.ncol - mid,
                self.ml.row_stride,
                self.ml.col_stride,
            ),
        )?;

        Ok((t0, t1))
    }
}

type Operator<T, M> = <<M as Memory<T>>::Base as MemoryOwn<T>>::Operator;

impl<T: ElemType, MM: MemoryMut<T>> Tensor<T, MM>
where
    Operator<T, MM>: BackendOps<T>,
{
    /*
    /// Adds `other` to `self` element-wise, supporting dynamic broadcasting.
    ///
    /// Conceptually:
    /// - Exact match: `self[i, j] += other[i, j]`
    /// - Broadcast: `self[i, j] += other[0, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if dimensions are incompatible for both exact match and broadcasting.
    pub fn add_assign<S0: Memory<T, Base = MM::Base>>(
        &mut self,
        other: &Tensor<T, S0>,
    ) -> Result<(), MLTError> {
        if self.ml.nrow == other.ml.nrow && self.ml.ncol == other.ml.ncol {
            unsafe {
                Operator::<T, MM>::elem_add_assign(&mut self.mem, &self.ml, &other.mem, &other.ml)
            };
        } else if self.ml.ncol == other.ml.ncol && other.ml.nrow == 1 {
            validate_major(true, self.ml.row_stride, self.ml.col_stride)?;
            validate_major(true, other.ml.row_stride, other.ml.col_stride)?;
            unsafe {
                Operator::<T, MM>::elem_br_add_assign(
                    &mut self.mem,
                    &self.ml,
                    &other.mem,
                    &other.ml,
                )
            };
        } else {
            return validate_shape(other.ml.nrow, other.ml.ncol, self.ml.nrow, self.ml.ncol);
        }
        Ok(())
    }

    /// Finds the index of the maximum value in the first row.
    ///
    /// Conceptually: Returns `i` such that `self[0, i] == max(self[0, 0..ncol])`.
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `self` does not have exactly 1 row.
    pub fn argmax(&self) -> Result<u32, MLTError> {
        validate_shape(self.ml.nrow, self.ml.ncol, 1, self.ml.ncol)?;
        validate_major(true, self.ml.row_stride, self.ml.col_stride)?;
        Ok(unsafe { Operator::<T, MM>::argmax(&self.mem, &self.ml) })
    }

    /// Copies values from `other` to `self`.
    ///
    /// Conceptually: `self[i, j] = other[i, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if dimensions are not strictly identical.
    pub fn copy<S0: Memory<T, Base = MM::Base>>(
        &mut self,
        other: &Tensor<T, S0>,
    ) -> Result<(), MLTError> {
        validate_shape(other.ml.nrow, other.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        unsafe { Operator::<T, MM>::copy(&mut self.mem, &self.ml, &other.mem, &other.ml) };
        Ok(())
    }

    /// Fills all elements of `self` with a given scalar value.
    ///
    /// Conceptually: `self[i, j] = value`
    pub fn fill(&mut self, value: T) {
        unsafe { Operator::<T, MM>::fill(&mut self.mem, &self.ml, value) };
    }

    /// Multiplies all elements of `self` by a given scalar value in-place.
    ///
    /// Conceptually: `self[i, j] *= value`
    pub fn scalar_mul_assign(&mut self, value: T) {
        unsafe { Operator::<T, MM>::scalar_mul_assign(&mut self.mem, &self.ml, value) };
    }

    /// Computes the element-wise product of `a` and `b` into `self`.
    ///
    /// Conceptually: `self[i, j] = a[i, j] * b[i, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `self`, `a`, or `b` do not share identical dimensions.
    pub fn elem_mul<S0: Memory<T, Base = MM::Base>, S1: Memory<T, Base = MM::Base>>(
        &mut self,
        a: &Tensor<T, S0>,
        b: &Tensor<T, S1>,
    ) -> Result<(), MLTError> {
        validate_shape(a.ml.nrow, a.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        validate_shape(b.ml.nrow, b.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        unsafe {
            Operator::<T, MM>::elem_mul(&mut self.mem, &self.ml, &a.mem, &a.ml, &b.mem, &b.ml)
        };
        Ok(())
    }

    /// Computes the element-wise product of `a` and `b` and adds it to `self`.
    ///
    /// Conceptually: `self[i, j] += a[i, j] * b[i, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `self`, `a`, or `b` do not share identical dimensions.
    pub fn elem_muladd_assign<S0: Memory<T, Base = MM::Base>, S1: Memory<T, Base = MM::Base>>(
        &mut self,
        a: &Tensor<T, S0>,
        b: &Tensor<T, S1>,
    ) -> Result<(), MLTError> {
        validate_shape(a.ml.nrow, a.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        validate_shape(b.ml.nrow, b.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        unsafe {
            Operator::<T, MM>::elem_muladd_assign(
                &mut self.mem,
                &self.ml,
                &a.mem,
                &a.ml,
                &b.mem,
                &b.ml,
            )
        };
        Ok(())
    }

    /// Computes the element-wise product of `a` and `b` and subtracts it from `self`.
    ///
    /// Conceptually: `self[i, j] -= a[i, j] * b[i, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `self`, `a`, or `b` do not share identical dimensions.
    pub fn elem_mulsub_assign<S0: Memory<T, Base = MM::Base>, S1: Memory<T, Base = MM::Base>>(
        &mut self,
        a: &Tensor<T, S0>,
        b: &Tensor<T, S1>,
    ) -> Result<(), MLTError> {
        validate_shape(a.ml.nrow, a.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        validate_shape(b.ml.nrow, b.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        unsafe {
            Operator::<T, MM>::elem_mulsub_assign(
                &mut self.mem,
                &self.ml,
                &a.mem,
                &a.ml,
                &b.mem,
                &b.ml,
            )
        };
        Ok(())
    }

    /// Multiplies `self` by `other` element-wise in-place.
    ///
    /// Conceptually: `self[i, j] *= other[i, j]`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if dimensions are not strictly identical.
    pub fn elem_mul_assign<S0: Memory<T, Base = MM::Base>>(
        &mut self,
        other: &Tensor<T, S0>,
    ) -> Result<(), MLTError> {
        validate_shape(other.ml.nrow, other.ml.ncol, self.ml.nrow, self.ml.ncol)?;
        unsafe {
            Operator::<T, MM>::elem_mul_assign(&mut self.mem, &self.ml, &other.mem, &other.ml)
        };
        Ok(())
    }

    /// Performs matrix multiplication of `a` and `b` into `self`.
    ///
    /// Conceptually: `self = a @ b`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if matrix dimensions are incompatible for multiplication.
    pub fn matmul<S0: Memory<T, Base = MM::Base>, S1: Memory<T, Base = MM::Base>>(
        &mut self,
        a: &Tensor<T, S0>,
        b: &Tensor<T, S1>,
    ) -> Result<(), MLTError> {
        validate_shape(self.ml.nrow, self.ml.ncol, a.ml.nrow, b.ml.ncol)?;
        validate_shape(b.ml.nrow, b.ml.ncol, a.ml.ncol, b.ml.ncol)?;
        unsafe { Operator::<T, MM>::matmul(&mut self.mem, &self.ml, &a.mem, &a.ml, &b.mem, &b.ml) };
        Ok(())
    }

    /// Performs in-place matrix multiplication of `self` and `other`.
    ///
    /// Conceptually: `self = self @ other`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `other` is not a square matrix, or if `other.nrow != self.ncol`.
    pub fn matmul_assign<S0: Memory<T, Base = MM::Base>>(
        &mut self,
        other: &Tensor<T, S0>,
    ) -> Result<(), MLTError> {
        validate_shape(other.ml.nrow, other.ml.ncol, self.ml.ncol, self.ml.ncol)?;
        unsafe { Operator::<T, MM>::matmul_assign(&mut self.mem, &self.ml, &other.mem, &other.ml) };
        Ok(())
    }

    /// Applies Root Mean Square (RMS) normalization to `self` and scales by `weight`.
    ///
    /// Conceptually: `self[i, j] = self[i, j] * weight[0, j] / (rms[i] + epsilon)`
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `weight` does not have shape `1 x self.ncol`.
    pub fn rms_norm<S0: Memory<T, Base = MM::Base>>(
        &mut self,
        weight: &Tensor<T, S0>,
        epsilon: f32,
    ) -> Result<(), MLTError> {
        validate_shape(weight.ml.nrow, weight.ml.ncol, 1, self.ml.ncol)?;
        validate_major(true, self.ml.row_stride, self.ml.col_stride)?;
        validate_major(true, weight.ml.row_stride, weight.ml.col_stride)?;
        unsafe {
            Operator::<T, MM>::rms_norm(&mut self.mem, &self.ml, &weight.mem, &weight.ml, epsilon)
        };
        Ok(())
    }

    /// Applies Rotary Position Embedding (RoPE) frequencies to `self`.
    ///
    /// Conceptually: Splits `self` along the column into halves, applying RoPE cosine
    /// to the first half and RoPE sine to the second half.
    ///
    /// # Errors
    /// Returns `MLTError::shape_mismatch` if `self` is not exactly `1 x head_size`.
    pub fn rope_vector(
        &mut self,
        token_index: u32,
        rope_theta: T,
        head_size: u32,
    ) -> Result<(), MLTError> {
        validate_shape(self.ml.nrow, self.ml.ncol, 1, head_size)?;
        validate_major(true, self.ml.row_stride, self.ml.col_stride)?;
        let (mut dst0, mut dst1) = self.split_col(head_size / 2)?;
        let k = T::from_f32(token_index as f32);
        let d = T::from_f32(head_size as f32);

        unsafe {
            Operator::<T, MM>::rope_cos(&mut dst0.mem, &dst0.ml, k, rope_theta, d);
            Operator::<T, MM>::rope_sin(&mut dst1.mem, &dst1.ml, k, rope_theta, d);
        };
        Ok(())
    }

    /// Applies the numerically safe softmax function to `self` in-place.
    ///
    /// Conceptually: `self[i, j] = exp(self[i, j] - max_j) / sum_k(exp(self[i, k] - max_k))`
    pub fn masked_safe_softmax(&mut self) -> Result<(), MLTError> {
        validate_major(true, self.ml.row_stride, self.ml.col_stride)?;
        for i in 0..self.ml.nrow {
            let n_mask = self.ml.nrow - 1 - i;
            let ml = self.ml.sliced(i..i + 1, 0..self.ml.ncol);
            unsafe { Operator::<T, MM>::masked_safe_softmax(&mut self.mem, &ml, n_mask) };
        }
        Ok(())
    }

    /// Applies the SiLU (Swish) activation function to `self` in-place.
    ///
    /// Conceptually: `self[i, j] = self[i, j] / (1 + exp(-self[i, j]))`
    pub fn silu(&mut self) {
        unsafe { Operator::<T, MM>::silu(&mut self.mem, &self.ml) };
    }
    */
}

fn validate_space<T>(mem_size: usize, ml: &MatrixLayout<T>) -> Result<(), MLTError> {
    if ml.nrow == 0 || ml.ncol == 0 {
        return Ok(());
    }
    let required_size = ((ml.nrow - 1) as usize * ml.row_stride as usize
        + (ml.ncol - 1) as usize * ml.col_stride as usize
        + 1)
        * size_of::<T>();
    if mem_size < required_size {
        return Err(MLTError::insufficient_storage_space(
            required_size,
            mem_size,
        ));
    }
    Ok(())
}

fn validate_shape(
    nrow_actual: u32,
    ncol_actual: u32,
    nrow_expected: u32,
    ncol_expected: u32,
) -> Result<(), MLTError> {
    if nrow_expected != nrow_actual {
        return Err(MLTError::shape_mismatch(
            nrow_expected as usize,
            nrow_actual as usize,
        ));
    }
    if ncol_expected != ncol_actual {
        return Err(MLTError::shape_mismatch(
            ncol_expected as usize,
            ncol_actual as usize,
        ));
    }
    Ok(())
}

fn validate_major(row_major: bool, row_stride: u32, col_stride: u32) -> Result<(), MLTError> {
    match row_major {
        true => {
            if col_stride != 1 {
                return Err(MLTError::matrix_layout_mismatch(true, false));
            }
        }
        false => {
            if row_stride != 1 {
                return Err(MLTError::matrix_layout_mismatch(false, true));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
impl<T: ElemType, M: Memory<T, Base = backend_host::Mmap<T>>> Tensor<T, M> {
    pub fn assert<const N: usize>(&self, answer: &[[T; N]]) {
        assert!(
            self.ml.nrow as usize == answer.len() && self.ml.ncol as usize == N,
            "invalid test data"
        );

        let mut ptr = unsafe { self.mem.as_ptr().byte_add(self.ml.offset) } as *const T;
        for i in 0..self.ml.nrow as usize {
            for j in 0..self.ml.ncol as usize {
                let expected = answer[i][j].to_f32();
                let actual = unsafe { *(ptr.add(self.ml.col_stride as usize * j)) }.to_f32();
                assert!(
                    (expected - actual).abs() < 0.0001,
                    "storage[{}][{}] mismatch: expected {:?}, actual {:?}",
                    i,
                    j,
                    expected,
                    actual,
                );
            }
            ptr = unsafe { ptr.add(self.ml.row_stride as usize) };
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use backend_host::Mmap;
    use core::{ElemType, MatrixLayout, MemoryMut, MemoryOwn};

    use super::Tensor;

    fn build_tensor<T: ElemType>(value: &[T], ml: MatrixLayout<T>) -> Tensor<T, Mmap<T>> {
        let mut mem = Mmap::new(value.len() * size_of::<T>()).expect("Failed to allocate Mmap");
        let dst = mem.as_mut_ptr();
        let src = value.as_ptr();
        let count = value.len();
        unsafe { std::ptr::copy_nonoverlapping(src, dst, count) };
        Tensor::new(mem, ml).expect("Failed to create tensor")
    }

    #[test]
    fn copy_subtensor() {
        let src = build_tensor(
            &[1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0],
            MatrixLayout::new(0, 2, 3, 3, 1),
        );
        let mut dst = build_tensor(
            &[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            MatrixLayout::new(0, 3, 4, 4, 1),
        );
        dst.slice_mut(1..3, 1..4)
            .copy(&src)
            .expect("Failed to copy tensor");
        dst.assert(&[
            [9.0, 9.0, 9.0, 9.0],
            [9.0, 1.0, 2.0, 3.0],
            [9.0, 2.0, 3.0, 4.0],
        ]);
    }

    #[test]
    fn rms_norm() {
        let mut x = build_tensor(&[3.0f32, 4.0, 0.0, 5.0], MatrixLayout::new(0, 2, 2, 2, 1));
        let w = build_tensor(&[2.0, 0.5], MatrixLayout::new(0, 1, 2, 2, 1));
        x.rms_norm(&w, 0.0).expect("Failed to RMS-normalize tensor");
        x.assert(&[[1.6970563, 0.56568545], [0.0, 0.70710677]]);
    }

    #[test]
    fn transpose() {
        let t = build_tensor(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            MatrixLayout::new(0, 2, 3, 3, 1),
        );
        t.transpose().assert(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn reshape() {
        let mut t = build_tensor(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            MatrixLayout::new(0, 2, 2, 2, 1),
        );
        t.reshape(3, 2, 2, 1).unwrap();
        t.assert(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    }

    #[test]
    fn split() {
        let mut t1 = build_tensor(&[1.0f32, 2.0, 3.0, 4.0], MatrixLayout::new(0, 2, 2, 2, 1));
        let (r0, r1) = t1.split_row(1).unwrap();
        r0.assert(&[[1.0, 2.0]]);
        r1.assert(&[[3.0, 4.0]]);

        let mut t2 = build_tensor(&[1.0f32, 2.0, 3.0, 4.0], MatrixLayout::new(0, 2, 2, 2, 1));
        let (c0, c1) = t2.split_col(1).unwrap();
        c0.assert(&[[1.0], [3.0]]);
        c1.assert(&[[2.0], [4.0]]);
    }

    #[test]
    fn add_assign() {
        let mut t1 = build_tensor(&[1.0f32, 2.0, 3.0, 4.0], MatrixLayout::new(0, 2, 2, 2, 1));
        let t2 = build_tensor(&[1.0f32, 1.0, 1.0, 1.0], MatrixLayout::new(0, 2, 2, 2, 1));

        t1.add_assign(&t2).unwrap();
        t1.assert(&[[2.0, 3.0], [4.0, 5.0]]);

        let t3 = build_tensor(&[10.0f32, 20.0], MatrixLayout::new(0, 1, 2, 2, 1));
        t1.add_assign(&t3).unwrap();
        t1.assert(&[[12.0, 23.0], [14.0, 25.0]]);
    }

    #[test]
    fn argmax() {
        let t = build_tensor(
            &[1.0f32, 5.0, 2.0, 4.0, 3.0],
            MatrixLayout::new(0, 1, 5, 5, 1),
        );
        assert_eq!(t.argmax().unwrap(), 1);
    }

    #[test]
    fn fill_and_scalar_mul() {
        let mut t = build_tensor(&[0.0f32; 4], MatrixLayout::new(0, 2, 2, 2, 1));
        t.fill(2.0);
        t.assert(&[[2.0, 2.0], [2.0, 2.0]]);

        t.scalar_mul_assign(3.0);
        t.assert(&[[6.0, 6.0], [6.0, 6.0]]);
    }

    #[test]
    fn matmul() {
        let mut c = build_tensor(&[0.0f32; 4], MatrixLayout::new(0, 2, 2, 2, 1));
        let a = build_tensor(&[1.0f32, 2.0, 3.0, 4.0], MatrixLayout::new(0, 2, 2, 2, 1));
        let b = build_tensor(&[2.0f32, 0.0, 1.0, 2.0], MatrixLayout::new(0, 2, 2, 2, 1));

        c.matmul(&a, &b).unwrap();
        c.assert(&[[4.0, 4.0], [10.0, 8.0]]);
    }

    #[test]
    fn activations() {
        let mut t1 = build_tensor(&[-5.0f32, 8.0, 3.0, 3.0], MatrixLayout::new(0, 2, 2, 2, 1));
        t1.masked_safe_softmax().expect("Failed to softmax");
        t1.assert(&[[1.0, 0.0], [0.5, 0.5]]);

        let mut t2 = build_tensor(&[0.0f32], MatrixLayout::new(0, 1, 1, 1, 1));
        t2.silu();
        t2.assert(&[[0.0]]);
    }
}
*/
