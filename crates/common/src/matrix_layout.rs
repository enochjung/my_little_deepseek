use crate::elem_type::ElemType;

use core::marker::PhantomData;
use core::ops::{Bound, RangeBounds};

pub struct MatrixLayout<T: ElemType> {
    pub offset: usize,
    pub nrow: u32,
    pub ncol: u32,
    pub row_stride: u32,
    pub col_stride: u32,
    _phantom: PhantomData<T>,
}

impl<T: ElemType> MatrixLayout<T> {
    pub fn new(offset: usize, nrow: u32, ncol: u32, row_stride: u32, col_stride: u32) -> Self {
        Self {
            offset,
            nrow,
            ncol,
            row_stride,
            col_stride,
            _phantom: PhantomData,
        }
    }

    pub fn rc_offset(&self, r: u32, c: u32) -> usize {
        (r as usize * self.row_stride as usize + c as usize * self.col_stride as usize)
            * size_of::<T>()
            + self.offset
    }

    pub fn transpose(mut self) -> Self {
        core::mem::swap(&mut self.nrow, &mut self.ncol);
        core::mem::swap(&mut self.row_stride, &mut self.col_stride);
        self
    }

    pub fn sliced(&self, rows: impl RangeBounds<u32>, cols: impl RangeBounds<u32>) -> Self {
        let (rs, re) = normalize(rows, self.nrow);
        let (cs, ce) = normalize(cols, self.ncol);

        Self {
            offset: self.rc_offset(rs, cs),
            nrow: re - rs,
            ncol: ce - cs,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
            _phantom: PhantomData,
        }
    }
}

fn normalize(range: impl RangeBounds<u32>, len: u32) -> (u32, u32) {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    (start.min(len), end.max(start).min(len))
}
