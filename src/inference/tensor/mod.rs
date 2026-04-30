use super::Error;
use crate::inference::{kernel, utils};
use core::marker::PhantomData;
use std::ops::Range;

pub trait DataType {
    const BYTES: usize;
}

pub struct BF16;
impl DataType for BF16 {
    const BYTES: usize = 2;
}

pub struct F32;
impl DataType for F32 {
    const BYTES: usize = 4;
}

pub trait StorageType {}

pub struct Host;
impl StorageType for Host {}

/// Tensor layout descriptor.
///
/// Fields:
/// - `is_row_major`: whether rows are contiguous in memory
/// - `nrow`: number of rows
/// - `ncol`: number of columns
/// - `stride`: number of elements between row starts
#[derive(Clone, Copy)]
pub struct Layout {
    pub is_row_major: bool,
    pub nrow: usize,
    pub ncol: usize,
    pub stride: usize,
}

/// Owned 2D tensor with aligned host storage.
pub struct Tensor<D: DataType, S: StorageType> {
    data: utils::AlignedBytes,
    layout: Layout,
    _pd: PhantomData<(D, S)>,
}

/// Borrowed immutable tensor view over bytes.
pub struct TensorRef<'a, D: DataType, S: StorageType> {
    data: &'a [u8],
    layout: Layout,
    _pd: PhantomData<(D, S)>,
}

/// Borrowed mutable tensor view over bytes.
pub struct TensorMut<'a, D: DataType, S: StorageType> {
    data: &'a mut [u8],
    layout: Layout,
    _pd: PhantomData<(D, S)>,
}

fn checked_total_bytes<D: DataType>(nrow: usize, ncol: usize) -> Result<usize, Error> {
    nrow.checked_mul(ncol)
        .and_then(|count| count.checked_mul(D::BYTES))
        .ok_or_else(|| Error::shape_mismatch(usize::MAX, 0))
}

fn validate_shape(nrow: usize, ncol: usize) -> Result<(), Error> {
    if nrow == 0 || ncol == 0 {
        return Err(Error::shape_mismatch(1, 0));
    }
    Ok(())
}

fn validate_layout<D: DataType>(
    is_row_major: bool,
    nrow: usize,
    ncol: usize,
    stride: usize,
    data_bytes: usize,
) -> Result<(), Error> {
    validate_shape(nrow, ncol)?;

    let min_stride = if is_row_major { ncol } else { nrow };
    if stride < min_stride {
        return Err(Error::shape_mismatch(min_stride, stride));
    }

    let required_bytes = if is_row_major {
        (nrow - 1)
            .checked_mul(stride)
            .and_then(|row_offset| row_offset.checked_add(ncol))
            .and_then(|element_count| element_count.checked_mul(D::BYTES))
    } else {
        (ncol - 1)
            .checked_mul(stride)
            .and_then(|col_offset| col_offset.checked_add(nrow))
            .and_then(|element_count| element_count.checked_mul(D::BYTES))
    }
    .ok_or_else(|| Error::shape_mismatch(0, 0))?;

    if data_bytes != required_bytes {
        return Err(Error::shape_mismatch(required_bytes, data_bytes));
    }

    Ok(())
}

fn validate_range(
    rows: &Range<usize>,
    cols: &Range<usize>,
    nrow: usize,
    ncol: usize,
) -> Result<(), Error> {
    if rows.end > nrow {
        return Err(Error::out_of_bound(rows.end, nrow));
    }
    if cols.end > ncol {
        return Err(Error::out_of_bound(cols.end, ncol));
    }
    Ok(())
}

fn subview_byte_range<D: DataType>(
    layout: &Layout,
    rows: &Range<usize>,
    cols: &Range<usize>,
) -> Range<usize> {
    let start_elem = if layout.is_row_major {
        rows.start * layout.stride + cols.start
    } else {
        cols.start * layout.stride + rows.start
    };

    let end_elem = if layout.is_row_major {
        (rows.end - 1) * layout.stride + cols.end
    } else {
        (cols.end - 1) * layout.stride + rows.end
    };

    let start = start_elem * D::BYTES;
    let end = end_elem * D::BYTES;

    if rows.start >= rows.end || cols.start >= cols.end {
        start..start
    } else {
        start..end
    }
}

impl<D: DataType, S: StorageType> Tensor<D, S> {
    /// Creates a non-initialized Tensor from layout fields.
    ///
    /// Parameters:
    /// - `is_row_major`: row-major flag.
    /// - `nrow`: number of rows.
    /// - `ncol`: number of columns.
    ///
    /// Returns:
    /// - `Result<Self, Error>`: new Tensor or error.
    ///
    /// Validation:
    /// - `nrow > 0` and `ncol > 0`.
    pub fn new(is_row_major: bool, nrow: usize, ncol: usize) -> Result<Self, Error> {
        validate_shape(nrow, ncol)?;

        let bytes = checked_total_bytes::<D>(nrow, ncol)?;
        let mut data = utils::AlignedBytes::with_capacity(bytes);
        data.set_len(bytes);

        let stride = if is_row_major { ncol } else { nrow };
        let layout = Layout {
            is_row_major,
            nrow,
            ncol,
            stride,
        };

        Ok(Self {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Preallocates an owned tensor with row-major for incremental append.
    ///
    /// Parameters:
    /// - `nrow_cap`: maximum row capacity to preallocate.
    /// - `ncol`: number of columns.
    ///
    /// Returns:
    /// - `Result<Self, Error>`: empty Tensor with reserved capacity or error.
    ///
    /// Validation:
    /// - `nrow_cap > 0` and `ncol > 0`.
    pub fn with_capacity(nrow_cap: usize, ncol: usize) -> Result<Self, Error> {
        validate_shape(nrow_cap, ncol)?;

        let bytes = checked_total_bytes::<D>(nrow_cap, ncol)?;
        let data = utils::AlignedBytes::with_capacity(bytes);
        let layout = Layout {
            is_row_major: true,
            nrow: 0,
            ncol,
            stride: ncol,
        };

        Ok(Self {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns a borrowed tensor view over a row and column subrange.
    ///
    /// Parameters:
    /// - `rows`: row range in `[start, end)` form.
    /// - `cols`: column range in `[start, end)` form.
    ///
    /// Returns:
    /// - `Result<TensorRef<'a, D, S>, Error>`: immutable sub-tensor view or error.
    ///
    /// Validation:
    /// - `rows.start <= rows.end <= nrow`.
    /// - `cols.start <= cols.end <= ncol`.
    pub fn as_ref<'a>(
        &'a self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<TensorRef<'a, D, S>, Error> {
        validate_range(&rows, &cols, self.layout.nrow, self.layout.ncol)?;

        let range = subview_byte_range::<D>(&self.layout, &rows, &cols);
        let data: &'a [u8] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_slice().as_ptr().add(range.start),
                range.end - range.start,
            )
        };
        let layout = Layout {
            is_row_major: self.layout.is_row_major,
            nrow: rows.end - rows.start,
            ncol: cols.end - cols.start,
            stride: self.layout.stride,
        };

        Ok(TensorRef {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns a writable tensor view over a row and column subrange.
    ///
    /// Parameters:
    /// - `rows`: row range in `[start, end)` form.
    /// - `cols`: column range in `[start, end)` form.
    ///
    /// Returns:
    /// - `Result<TensorMut<'a, D, S>, Error>`: writable sub-tensor view or error.
    ///
    /// Validation:
    /// - `rows.start <= rows.end <= nrow`.
    /// - `cols.start <= cols.end <= ncol`.
    pub fn as_mut<'a>(
        &'a mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<TensorMut<'a, D, S>, Error> {
        validate_range(&rows, &cols, self.layout.nrow, self.layout.ncol)?;

        let range = subview_byte_range::<D>(&self.layout, &rows, &cols);
        let data: &'a mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_slice().as_mut_ptr().add(range.start),
                range.end - range.start,
            )
        };
        let layout = Layout {
            is_row_major: self.layout.is_row_major,
            nrow: rows.end - rows.start,
            ncol: cols.end - cols.start,
            stride: self.layout.stride,
        };

        Ok(TensorMut {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns the tensor with transposed layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `TensorRef<'a, D, S>`: transposed tensor view.
    ///
    /// Validation:
    /// - None
    pub fn transpose<'a>(&'a self) -> TensorRef<'a, D, S> {
        let layout = Layout {
            is_row_major: !self.layout.is_row_major,
            nrow: self.layout.ncol,
            ncol: self.layout.nrow,
            stride: self.layout.stride,
        };

        TensorRef {
            data: self.data.as_slice(),
            layout,
            _pd: PhantomData,
        }
    }

    /// Returns the tensor layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `&Layout`: reference to the tensor layout.
    ///
    /// Validation:
    /// - None
    pub fn layout<'a>(&'a self) -> &'a Layout {
        &self.layout
    }

    /// Returns pointer to first element of tensor data (const).
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*const u8`: pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    fn data(&self) -> *const u8 {
        self.data.as_slice().as_ptr()
    }

    /// Returns mutable pointer to first element of tensor data.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*mut u8`: mutable pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    fn data_mut(&mut self) -> *mut u8 {
        self.data.as_mut_slice().as_mut_ptr()
    }

    /// Appends rows from `other` into `self`.
    ///
    /// Validation:
    /// - Both tensors must be row-major contiguous and have matching `ncol`.
    pub fn append(&mut self, other: &TensorRef<'_, D, S>) -> Result<(), Error> {
        match (self.layout.is_row_major, other.layout.is_row_major) {
            (true, true) => {
                if self.layout.ncol != other.layout.ncol {
                    return Err(Error::shape_mismatch(self.layout.ncol, other.layout.ncol));
                }

                if other.layout.stride == self.layout.stride {
                    self.data.extend_from_slice(other.data);
                } else {
                    let row_bytes = self.layout.ncol * D::BYTES;
                    for row in 0..other.layout.nrow {
                        let row_start = row * other.layout.stride * D::BYTES;
                        let row_end = row_start + row_bytes;
                        self.data.extend_from_slice(&other.data[row_start..row_end]);
                    }
                }

                Ok(())
            }
            (false, false) => {
                if self.layout.nrow != other.layout.nrow {
                    return Err(Error::shape_mismatch(self.layout.nrow, other.layout.nrow));
                }

                if other.layout.stride == self.layout.stride {
                    self.data.extend_from_slice(other.data);
                } else {
                    let col_bytes = self.layout.nrow * D::BYTES;
                    for col in 0..other.layout.ncol {
                        let col_start = col * other.layout.stride * D::BYTES;
                        let col_end = col_start + col_bytes;
                        self.data.extend_from_slice(&other.data[col_start..col_end]);
                    }
                }

                Ok(())
            }
            (true, false) => {
                unimplemented!()
            }
            (false, true) => {
                unimplemented!()
            }
        }
    }
}

impl<'a, D: DataType, S: StorageType> TensorRef<'a, D, S> {
    /// Constructs a tensor from a byte slice and layout fields.
    ///
    /// Parameters:
    /// - `data`: byte slice containing tensor storage.
    /// - `is_row_major`: row-major flag.
    /// - `nrow`: number of rows.
    /// - `ncol`: number of columns.
    /// - `stride`: row stride.
    ///
    /// Returns:
    /// - `Result<Self, Error>`: new `TensorRef` or error.
    pub fn new(
        data: &'a [u8],
        is_row_major: bool,
        nrow: usize,
        ncol: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate_shape(nrow, ncol)?;
        validate_layout::<D>(is_row_major, nrow, ncol, stride, data.len())?;

        let layout = Layout {
            is_row_major,
            nrow,
            ncol,
            stride,
        };

        Ok(Self {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns an immutable subview of this `TensorRef`.
    ///
    /// Parameters:
    /// - `rows`: row range in `[start, end)` form.
    /// - `cols`: column range in `[start, end)` form.
    ///
    /// Returns:
    /// - `Result<TensorRef<'a, D, S>, Error>`: immutable sub-tensor view or error.
    ///
    /// Validation:
    /// - `rows.start <= rows.end <= nrow`.
    /// - `cols.start <= cols.end <= ncol`.
    pub fn as_ref(
        &self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<TensorRef<'a, D, S>, Error> {
        validate_range(&rows, &cols, self.layout.nrow, self.layout.ncol)?;

        let range = subview_byte_range::<D>(&self.layout, &rows, &cols);
        let data = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr().add(range.start), range.end - range.start)
        };
        let layout = Layout {
            is_row_major: self.layout.is_row_major,
            nrow: rows.end - rows.start,
            ncol: cols.end - cols.start,
            stride: self.layout.stride,
        };

        Ok(TensorRef {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns the tensor with transposed layout (borrowing same bytes).
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `TensorRef<'a, D, S>`: tensor with `nrow` and `ncol` swapped.
    ///
    /// Validation:
    /// - None
    pub fn transpose(&self) -> TensorRef<'a, D, S> {
        let layout = Layout {
            is_row_major: !self.layout.is_row_major,
            nrow: self.layout.ncol,
            ncol: self.layout.nrow,
            stride: self.layout.stride,
        };

        TensorRef {
            data: self.data,
            layout,
            _pd: PhantomData,
        }
    }

    /// Returns the tensor layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `&Layout`: reference to the tensor layout.
    ///
    /// Validation:
    /// - None
    pub fn layout<'b>(&'b self) -> &'b Layout {
        &self.layout
    }

    /// Returns pointer to first element of tensor data (const).
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*const u8`: pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    pub fn data(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

impl<'a, D: DataType, S: StorageType> TensorMut<'a, D, S> {
    /// Constructs a writable tensor from a mutable byte slice and layout fields.
    ///
    /// Parameters:
    /// - `data`: mutable byte slice containing tensor storage.
    /// - `is_row_major`: row-major flag.
    /// - `nrow`: number of rows.
    /// - `ncol`: number of columns.
    /// - `stride`: row stride.
    ///
    /// Returns:
    /// - `Result<Self, Error>`: new `TensorMut` or error.
    pub fn new(
        data: &'a mut [u8],
        is_row_major: bool,
        nrow: usize,
        ncol: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate_layout::<D>(is_row_major, nrow, ncol, stride, data.len())?;

        let layout = Layout {
            is_row_major,
            nrow,
            ncol,
            stride,
        };

        Ok(Self {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Splits the tensor by rows into two writable views.
    ///
    /// Divides the current tensor into two `TensorMut` views at row index `mid`.
    /// Returns two tensors with rows `[0..mid)` and `[mid..nrow)` respectively,
    /// preserving layout and stride.
    ///
    /// Parameters:
    /// - `mid`: row index at which to split.
    ///
    /// Returns:
    /// - `Result<(Self, Self), Error>`: upper and lower mutable tensor views or error.
    ///
    /// Validation:
    /// - `0 < mid < nrow` (zero-sized halves are disallowed).
    pub fn split_row(self, mid: usize) -> Result<(Self, Self), Error> {
        if mid == 0 || mid >= self.layout.nrow {
            return Err(Error::out_of_bound(mid, self.layout.nrow));
        }

        let upper_range =
            subview_byte_range::<D>(&self.layout, &(0usize..mid), &(0usize..self.layout.ncol));
        let lower_range = subview_byte_range::<D>(
            &self.layout,
            &(mid..self.layout.nrow),
            &(0usize..self.layout.ncol),
        );

        let upper_data = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(upper_range.start),
                upper_range.end - upper_range.start,
            )
        };

        let lower_data = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(lower_range.start),
                lower_range.end - lower_range.start,
            )
        };

        Ok((
            Self {
                data: upper_data,
                layout: Layout {
                    is_row_major: self.layout.is_row_major,
                    nrow: mid,
                    ncol: self.layout.ncol,
                    stride: self.layout.stride,
                },
                _pd: PhantomData,
            },
            Self {
                data: lower_data,
                layout: Layout {
                    is_row_major: self.layout.is_row_major,
                    nrow: self.layout.nrow - mid,
                    ncol: self.layout.ncol,
                    stride: self.layout.stride,
                },
                _pd: PhantomData,
            },
        ))
    }

    /// Splits the tensor by columns into two writable views.
    ///
    /// Divides the current tensor into two `TensorMut` views at column index `mid`.
    /// Returns two tensors with columns `[0..mid)` and `[mid..ncol)` respectively,
    /// preserving layout and stride.
    ///
    /// Parameters:
    /// - `mid`: column index at which to split.
    ///
    /// Returns:
    /// - `Result<(Self, Self), Error>`: left and right mutable tensor views or error.
    ///
    /// Validation:
    /// - `0 < mid < ncol` (zero-sized halves are disallowed).
    pub fn split_col(self, mid: usize) -> Result<(Self, Self), Error> {
        if mid == 0 || mid >= self.layout.ncol {
            return Err(Error::out_of_bound(mid, self.layout.ncol));
        }

        let left_range =
            subview_byte_range::<D>(&self.layout, &(0usize..self.layout.nrow), &(0usize..mid));
        let right_range = subview_byte_range::<D>(
            &self.layout,
            &(0usize..self.layout.nrow),
            &(mid..self.layout.ncol),
        );

        let left_data = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(left_range.start),
                left_range.end - left_range.start,
            )
        };

        let right_data = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(right_range.start),
                right_range.end - right_range.start,
            )
        };

        Ok((
            Self {
                data: left_data,
                layout: Layout {
                    is_row_major: self.layout.is_row_major,
                    nrow: self.layout.nrow,
                    ncol: mid,
                    stride: self.layout.stride,
                },
                _pd: PhantomData,
            },
            Self {
                data: right_data,
                layout: Layout {
                    is_row_major: self.layout.is_row_major,
                    nrow: self.layout.nrow,
                    ncol: self.layout.ncol - mid,
                    stride: self.layout.stride,
                },
                _pd: PhantomData,
            },
        ))
    }

    /// Returns a writable subview of this `TensorMut`.
    ///
    /// Parameters:
    /// - `rows`: row range in `[start, end)` form.
    /// - `cols`: column range in `[start, end)` form.
    ///
    /// Returns:
    /// - `Result<TensorMut<'a, D, S>, Error>`: writable sub-tensor view or error.
    ///
    /// Validation:
    /// - `rows.start <= rows.end <= nrow`.
    /// - `cols.start <= cols.end <= ncol`.
    pub fn as_mut(
        &mut self,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Result<TensorMut<'a, D, S>, Error> {
        validate_range(&rows, &cols, self.layout.nrow, self.layout.ncol)?;

        let range = subview_byte_range::<D>(&self.layout, &rows, &cols);
        let data = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(range.start),
                range.end - range.start,
            )
        };
        let layout = Layout {
            is_row_major: self.layout.is_row_major,
            nrow: rows.end - rows.start,
            ncol: cols.end - cols.start,
            stride: self.layout.stride,
        };

        Ok(TensorMut {
            data,
            layout,
            _pd: PhantomData,
        })
    }

    /// Returns the tensor with transposed layout (mutable view).
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `TensorMut<'a, D, S>`: tensor with `nrow` and `ncol` swapped.
    ///
    /// Validation:
    /// - None
    pub fn transpose(&mut self) -> TensorMut<'a, D, S> {
        let data =
            unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr(), self.data.len()) };
        let layout = Layout {
            is_row_major: !self.layout.is_row_major,
            nrow: self.layout.ncol,
            ncol: self.layout.nrow,
            stride: self.layout.stride,
        };

        TensorMut {
            data,
            layout,
            _pd: PhantomData,
        }
    }

    /// Returns the tensor layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `&Layout`: reference to the tensor layout.
    ///
    /// Validation:
    /// - None
    pub fn layout<'b>(&'b self) -> &'b Layout {
        &self.layout
    }

    /// Returns mutable pointer to first element of tensor data.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `*mut u8`: mutable pointer to first element from storage.
    ///
    /// Validation:
    /// - None
    pub fn data_mut(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
}

impl<D: DataType, S: StorageType + Clone> Clone for Tensor<D, S> {
    /// Clones `Tensor` by cloning storage and layout.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - `Self`: cloned tensor.
    ///
    /// Validation:
    /// - None
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            layout: self.layout,
            _pd: PhantomData,
        }
    }
}

impl<'a> From<&TensorRef<'a, BF16, Host>> for Tensor<F32, Host> {
    /// Converts a `BF16` `TensorRef` to an owned `F32` `Tensor`.
    ///
    /// Parameters:
    /// - `src`: BF16 `TensorRef` source.
    ///
    /// Returns:
    /// - `Self`: newly allocated F32 `Tensor`.
    ///
    /// Validation:
    /// - None
    fn from(src: &TensorRef<'a, BF16, Host>) -> Self {
        let mut dst = Self::new(src.layout.is_row_major, src.layout.nrow, src.layout.ncol)
            .expect("valid BF16 to F32 conversion layout should allocate");
        let src_bytes = src.data;
        let dst_bytes = dst.data.as_mut_slice();

        match src.layout.is_row_major {
            true => {
                for row in 0..src.layout.nrow {
                    for col in 0..src.layout.ncol {
                        let src_elem = row * src.layout.stride + col;
                        let src_offset = src_elem * BF16::BYTES;
                        let bf16_bits =
                            u16::from_le_bytes([src_bytes[src_offset], src_bytes[src_offset + 1]]);
                        let value = f32::from_bits((bf16_bits as u32) << 16);

                        let dst_elem = row * dst.layout.stride + col;
                        let dst_offset = dst_elem * F32::BYTES;
                        dst_bytes[dst_offset..(dst_offset + 4)]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
            false => {
                for col in 0..src.layout.ncol {
                    for row in 0..src.layout.nrow {
                        let src_elem = col * src.layout.stride + row;
                        let src_offset = src_elem * BF16::BYTES;
                        let bf16_bits =
                            u16::from_le_bytes([src_bytes[src_offset], src_bytes[src_offset + 1]]);
                        let value = f32::from_bits((bf16_bits as u32) << 16);

                        let dst_elem = col * dst.layout.stride + row;
                        let dst_offset = dst_elem * F32::BYTES;
                        dst_bytes[dst_offset..(dst_offset + 4)]
                            .copy_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }

        dst
    }
}

impl Tensor<F32, Host> {
    /// Applies SiLU (Sigmoid Linear Unit) activation in place.
    ///
    /// Parameters:
    /// - None
    ///
    /// Returns:
    /// - None
    ///
    /// Validation:
    /// - None
    pub fn silu(&mut self) -> () {
        let x = self.data.as_mut_slice().as_mut_ptr() as *mut f32;
        let n = self.layout.nrow * self.layout.ncol;

        unsafe { kernel::x86_64::silu(x, n) };
    }

    /// Applies RMS normalization in place.
    ///
    /// Parameters:
    /// - `weight`: `TensorRef` with `nrow` 1 and `ncol` equal to the tensor `ncol`.
    /// - `epsilon`: small constant for numerical stability.
    ///
    /// Returns:
    /// - `Result<(), Error>`: success or error.
    ///
    /// Validation:
    /// - Writable contiguous layout required; `weight` `ncol` must match.
    pub fn rms_norm(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        epsilon: f32,
    ) -> Result<(), Error> {
        match self.layout.is_row_major {
            true => {
                if self.layout.ncol != weight.layout.ncol {
                    return Err(Error::shape_mismatch(self.layout.ncol, weight.layout.ncol));
                }

                let ncol = self.layout.ncol;
                let nrow = self.layout.nrow;

                let data_ptr = self.data.as_mut_slice().as_mut_ptr() as *mut f32;
                let weight_ptr = weight.data.as_ptr() as *const f32;

                for _ in 0..nrow {
                    let row_rms = unsafe { kernel::x86_64::rms(data_ptr, ncol) };
                    let scale = 1.0 / ((row_rms * row_rms) + epsilon).sqrt();
                    unsafe {
                        kernel::x86_64::mul(data_ptr, weight_ptr, scale, ncol);
                        data_ptr.add(ncol);
                    };
                }

                Ok(())
            }
            false => {
                // Column-major self path is not implemented yet.
                unimplemented!()
            }
        }
    }

    /// Computes self = self * weight + bias.
    ///
    /// Parameters:
    /// - `weight`: `TensorRef` with `nrow` and `ncol` both equal to the tensor `ncol`.
    /// - `bias`: `TensorRef` with `nrow` 1 and `ncol` equal to the tensor `ncol`.
    ///
    /// Returns:
    /// - `Result<(), Error>`: success or error.
    ///
    /// Validation:
    /// - Writable contiguous layout required; layout fields must match the expected dimensions.
    pub fn muladd_weight_bias(
        &mut self,
        weight: &TensorRef<'_, F32, Host>,
        bias: &TensorRef<'_, F32, Host>,
    ) -> Result<(), Error> {
        let expected = checked_total_bytes::<F32>(self.layout.nrow, self.layout.ncol)?;
        if !self.layout.is_row_major || self.layout.stride != self.layout.ncol {
            return Err(Error::shape_mismatch(expected, self.data.as_slice().len()));
        }
        if self.data.as_slice().len() != expected {
            return Err(Error::shape_mismatch(expected, self.data.as_slice().len()));
        }
        if weight.layout.nrow != self.layout.ncol || weight.layout.ncol != self.layout.ncol {
            return Err(Error::shape_mismatch(
                self.layout.ncol * self.layout.ncol * F32::BYTES,
                weight.data.len(),
            ));
        }
        if bias.layout.nrow != 1 || bias.layout.ncol != self.layout.ncol {
            return Err(Error::shape_mismatch(
                self.layout.ncol * F32::BYTES,
                bias.data.len(),
            ));
        }

        validate_layout::<F32>(
            weight.layout.is_row_major,
            weight.layout.nrow,
            weight.layout.ncol,
            weight.layout.stride,
            weight.data.len(),
        )?;
        validate_layout::<F32>(
            bias.layout.is_row_major,
            bias.layout.nrow,
            bias.layout.ncol,
            bias.layout.stride,
            bias.data.len(),
        )?;

        let nrow = self.layout.nrow;
        let ncol = self.layout.ncol;
        let values_ptr = self.data.as_mut_slice().as_mut_ptr() as *mut f32;
        let weight_ptr = weight.data.as_ptr() as *const f32;
        let bias_ptr = bias.data.as_ptr() as *const f32;
        let mut buf = vec![0f32; ncol];
        unsafe {
            kernel::x86_64::muladd_mn_nn_1n(
                values_ptr,
                self.layout.is_row_major,
                self.layout.stride,
                weight_ptr,
                weight.layout.is_row_major,
                weight.layout.stride,
                bias_ptr,
                buf.as_mut_ptr(),
                nrow,
                ncol,
            )
        };

        Ok(())
    }
}
