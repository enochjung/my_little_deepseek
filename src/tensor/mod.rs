mod numeric;

use crate::kernel::x86_64::cast_bf16_to_f32;
use crate::storage::*;
use core::marker::PhantomData;
pub(crate) use numeric::*;
use std::ops::Range;

pub(crate) struct Layout<N: Numeric> {
    offset: usize,
    is_row_major: bool,
    nrow: u32,
    ncol: u32,
    stride: u32,
    _phantom: PhantomData<N>,
}

impl<N: Numeric> Layout<N> {
    fn new(offset: usize, is_row_major: bool, nrow: u32, ncol: u32, stride: u32) -> Self {
        Self {
            offset,
            is_row_major,
            nrow,
            ncol,
            stride,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn shape(&self) -> [u32; 2] {
        [self.nrow, self.ncol]
    }

    fn validate_range(range: Range<u32>, limit: u32) -> Result<(u32, u32), crate::Error> {
        if range.start > range.end || range.end > limit {
            return Err(crate::Error::out_of_bound(
                range.end as usize,
                limit as usize,
            ));
        }

        Ok((range.start, range.end))
    }

    fn required_bytes(&self) -> usize {
        if self.nrow == 0 || self.ncol == 0 {
            return self.offset;
        }

        let elements = if self.is_row_major {
            (self.nrow as usize - 1) * self.stride as usize + self.ncol as usize
        } else {
            (self.ncol as usize - 1) * self.stride as usize + self.nrow as usize
        };

        self.offset + elements * N::BYTES
    }

    fn row_offset(&self, row: u32, col: u32) -> usize {
        let element_index = if self.is_row_major {
            row as usize * self.stride as usize + col as usize
        } else {
            col as usize * self.stride as usize + row as usize
        };

        self.offset + element_index * N::BYTES
    }

    fn slice(&self, rows: Range<u32>, cols: Range<u32>) -> Result<Self, crate::Error> {
        let (row_start, row_end) = Self::validate_range(rows, self.nrow)?;
        let (col_start, col_end) = Self::validate_range(cols, self.ncol)?;

        let offset = self.row_offset(row_start, col_start);
        let nrow = row_end - row_start;
        let ncol = col_end - col_start;

        Ok(Self::new(
            offset,
            self.is_row_major,
            nrow,
            ncol,
            self.stride,
        ))
    }
}

// Always packed (stride == ncol when row-major, vice versa)
pub(crate) struct TensorOwn<N: Numeric, S: Storage> {
    storage: S,
    layout: Layout<N>,
}

pub(crate) struct TensorRef<'a, N: Numeric, S: Storage> {
    storage: &'a S,
    layout: Layout<N>,
}

pub(crate) struct TensorMut<'a, N: Numeric, S: Storage> {
    storage: &'a mut S,
    layout: Layout<N>,
}

pub(crate) trait Tensor<N: Numeric, S: Storage> {
    fn storage(&self) -> &S;
    fn layout(&self) -> &Layout<N>;
    fn layout_mut(&mut self) -> &mut Layout<N>;

    fn shape(&self) -> [u32; 2] {
        self.layout().shape()
    }

    fn as_ref(
        &self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorRef<'_, N, S>, crate::Error> {
        let sliced_layout = self.layout().slice(rows, cols)?;
        TensorRef::new(
            self.storage(),
            sliced_layout.offset,
            sliced_layout.is_row_major,
            sliced_layout.nrow,
            sliced_layout.ncol,
            sliced_layout.stride,
        )
    }

    fn as_ptr(&self) -> *const u8 {
        self.storage().as_ptr().wrapping_add(self.layout().offset)
    }

    fn transpose(mut self) -> Self
    where
        Self: Sized,
    {
        let layout = self.layout_mut();
        layout.is_row_major = !layout.is_row_major;
        core::mem::swap(&mut layout.nrow, &mut layout.ncol);
        self
    }
}

pub(crate) trait MutableTensor<N: Numeric, S: Storage>: Tensor<N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error>;
    fn as_mut_ptr(&mut self) -> Result<*mut u8, crate::Error>;
}

impl<N: Numeric, S: Storage> Tensor<N, S> for TensorOwn<N, S> {
    fn storage(&self) -> &S {
        &self.storage
    }

    fn layout(&self) -> &Layout<N> {
        &self.layout
    }

    fn layout_mut(&mut self) -> &mut Layout<N> {
        &mut self.layout
    }
}

impl<'a, N: Numeric, S: Storage> Tensor<N, S> for TensorRef<'a, N, S> {
    fn storage(&self) -> &S {
        self.storage
    }

    fn layout(&self) -> &Layout<N> {
        &self.layout
    }

    fn layout_mut(&mut self) -> &mut Layout<N> {
        &mut self.layout
    }
}

impl<'a, N: Numeric, S: Storage> Tensor<N, S> for TensorMut<'a, N, S> {
    fn storage(&self) -> &S {
        self.storage
    }

    fn layout(&self) -> &Layout<N> {
        &self.layout
    }

    fn layout_mut(&mut self) -> &mut Layout<N> {
        &mut self.layout
    }
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for TensorOwn<N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error> {
        let sliced_layout = self.layout.slice(rows, cols)?;
        TensorMut::new(
            &mut self.storage,
            sliced_layout.offset,
            sliced_layout.is_row_major,
            sliced_layout.nrow,
            sliced_layout.ncol,
            sliced_layout.stride,
        )
    }

    fn as_mut_ptr(&mut self) -> Result<*mut u8, crate::Error> {
        Ok(self.storage.as_mut_ptr()?.wrapping_add(self.layout.offset) as *mut u8)
    }
}

impl<'a, N: Numeric, S: Storage> MutableTensor<N, S> for TensorMut<'a, N, S> {
    fn as_mut(
        &mut self,
        rows: Range<u32>,
        cols: Range<u32>,
    ) -> Result<TensorMut<'_, N, S>, crate::Error> {
        let sliced_layout = self.layout.slice(rows, cols)?;
        TensorMut::new(
            &mut *self.storage,
            sliced_layout.offset,
            sliced_layout.is_row_major,
            sliced_layout.nrow,
            sliced_layout.ncol,
            sliced_layout.stride,
        )
    }

    fn as_mut_ptr(&mut self) -> Result<*mut u8, crate::Error> {
        Ok(self.storage.as_mut_ptr()?.wrapping_add(self.layout.offset) as *mut u8)
    }
}

impl<N: Numeric, S: Storage> TensorOwn<N, S> {
    pub(crate) fn new(
        storage: S,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
    ) -> Result<Self, crate::Error> {
        let stride = if is_row_major { ncol } else { nrow };
        let layout = Layout::new(0, is_row_major, nrow, ncol, stride);
        let required = layout.required_bytes();
        let actual = storage.len();

        if actual < required {
            return Err(crate::Error::shape_mismatch(required, actual));
        }

        Ok(Self { storage, layout })
    }

    pub(crate) fn append<T: Tensor<N, S>>(&mut self, other: &T) -> Result<(), crate::Error> {
        let other_shape = other.shape();
        if other_shape[0] == 0 || other_shape[1] == 0 {
            return Ok(());
        }

        if self.layout.is_row_major != other.layout().is_row_major {
            return Err(crate::Error::operation_not_supported(
                "append with mismatched tensor layout majority (row-major vs col-major)",
            ));
        }

        if self.layout.is_row_major {
            let appended_rows = other_shape[0];

            if self.layout.ncol != other_shape[1] {
                return Err(crate::Error::shape_mismatch(
                    self.layout.ncol as usize,
                    other_shape[1] as usize,
                ));
            }

            let row_bytes = other_shape[1] as usize * N::BYTES;
            let src_stride = other.layout().stride as usize;
            self.append_lines(other, appended_rows as usize, row_bytes, src_stride);

            self.layout.nrow += appended_rows;
        } else {
            let appended_cols = other_shape[1];

            if self.layout.nrow != other_shape[0] {
                return Err(crate::Error::shape_mismatch(
                    self.layout.nrow as usize,
                    other_shape[0] as usize,
                ));
            }

            let col_bytes = other_shape[0] as usize * N::BYTES;
            let src_stride = other.layout().stride as usize;
            self.append_lines(other, appended_cols as usize, col_bytes, src_stride);

            self.layout.ncol += appended_cols;
        }

        Ok(())
    }

    fn append_lines<T: Tensor<N, S>>(
        &mut self,
        other: &T,
        lines: usize,
        line_bytes: usize,
        src_stride: usize,
    ) -> () {
        let dst_offset = self.layout.nrow as usize * self.layout.ncol as usize * N::BYTES;
        let src_stride_bytes = src_stride * N::BYTES;

        if line_bytes == src_stride_bytes {
            let copy_len = lines * line_bytes;
            self.storage
                .memory_copy(dst_offset, other.storage(), other.layout().offset, copy_len)
                .expect("append packed copy should succeed");
        } else {
            for idx in 0..lines {
                let dst_line_offset = dst_offset + idx * line_bytes;
                let src_line_offset = other.layout().offset + idx * src_stride_bytes;
                self.storage
                    .memory_copy(
                        dst_line_offset,
                        other.storage(),
                        src_line_offset,
                        line_bytes,
                    )
                    .expect("append strided copy should succeed");
            }
        }
    }
}

impl<'a, N: Numeric, S: Storage> TensorRef<'a, N, S> {
    pub(crate) fn new(
        storage: &'a S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<Self, crate::Error> {
        let layout = Layout::new(offset, is_row_major, nrow, ncol, stride);
        let required = layout.required_bytes();
        let actual = storage.len();

        if actual < required {
            return Err(crate::Error::shape_mismatch(required, actual));
        }

        Ok(Self { storage, layout })
    }
}

impl<'a, N: Numeric, S: Storage> TensorMut<'a, N, S> {
    pub(crate) fn new(
        storage: &'a mut S,
        offset: usize,
        is_row_major: bool,
        nrow: u32,
        ncol: u32,
        stride: u32,
    ) -> Result<Self, crate::Error> {
        let layout = Layout::new(offset, is_row_major, nrow, ncol, stride);
        let required = layout.required_bytes();
        let actual = storage.len();

        if actual < required {
            return Err(crate::Error::shape_mismatch(required, actual));
        }

        Ok(Self { storage, layout })
    }

    pub(crate) fn split_row(self, mid: usize) -> Result<(Self, Self), crate::Error> {
        let Self { storage, layout } = self;
        if mid > layout.nrow as usize {
            return Err(crate::Error::out_of_bound(mid, layout.nrow as usize));
        }

        let storage_ptr = storage as *mut S;
        let first = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout::new(
                layout.offset,
                layout.is_row_major,
                mid as u32,
                layout.ncol,
                layout.stride,
            ),
        };

        let second_offset = if layout.is_row_major {
            layout.offset + mid * layout.stride as usize * N::BYTES
        } else {
            layout.offset + mid * N::BYTES
        };
        let second = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout::new(
                second_offset,
                layout.is_row_major,
                layout.nrow - mid as u32,
                layout.ncol,
                layout.stride,
            ),
        };

        Ok((first, second))
    }

    pub(crate) fn split_col(self, mid: usize) -> Result<(Self, Self), crate::Error> {
        let Self { storage, layout } = self;
        if mid > layout.ncol as usize {
            return Err(crate::Error::out_of_bound(mid, layout.ncol as usize));
        }

        let storage_ptr = storage as *mut S;
        let first = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout::new(
                layout.offset,
                layout.is_row_major,
                layout.nrow,
                mid as u32,
                layout.stride,
            ),
        };

        let second_offset = if layout.is_row_major {
            layout.offset + mid * N::BYTES
        } else {
            layout.offset + mid * layout.stride as usize * N::BYTES
        };
        let second = Self {
            storage: unsafe { &mut *storage_ptr },
            layout: Layout::new(
                second_offset,
                layout.is_row_major,
                layout.nrow,
                layout.ncol - mid as u32,
                layout.stride,
            ),
        };

        Ok((first, second))
    }
}

impl<N: Numeric> Clone for TensorOwn<N, Host> {
    fn clone(&self) -> Self {
        let storage_len = self.storage.len();
        let name = format!("{} (cloned)", self.storage.name());

        let mut cloned_storage =
            Host::new(&name, storage_len).expect("cloning tensor storage should succeed");
        cloned_storage
            .memory_copy(0, &self.storage, 0, storage_len)
            .expect("copying tensor data to cloned storage should succeed");

        Self {
            storage: cloned_storage,
            layout: Layout::new(
                self.layout.offset,
                self.layout.is_row_major,
                self.layout.nrow,
                self.layout.ncol,
                self.layout.stride,
            ),
        }
    }
}

impl<'a, T: Tensor<BF16, Host>> From<&T> for TensorOwn<F32, Host> {
    fn from(src: &T) -> Self {
        let name = format!("{} (f32)", src.storage().name());

        let [nrow, ncol] = src.shape();
        let total = nrow as usize * ncol as usize;
        let storage = Host::new(&name, total * F32::BYTES)
            .expect("creating host storage for bf16->f32 conversion should succeed");

        unsafe { cast_bf16_to_f32(storage.as_ptr() as *mut f32, src.as_ptr(), total) };

        TensorOwn::new(storage, true, nrow, ncol)
            .expect("converted bf16 tensor should fit in its destination storage")
    }
}

impl TensorOwn<F32, Host> {
    pub(crate) fn silu(&mut self) -> () {
        todo!()
    }

    pub(crate) fn rms_norm(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        epsilon: f32,
    ) -> Result<(), crate::Error> {
        todo!()
    }

    pub(crate) fn muladd_weight_bias(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        bias: &TensorRef<'_, F32, Host>,
    ) -> Result<(), crate::Error> {
        todo!()
    }
}
