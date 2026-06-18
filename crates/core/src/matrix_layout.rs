use std::marker::PhantomData;
use std::ops::Range;

pub struct MatrixLayout<T> {
    pub offset: usize,
    pub nrow: u32,
    pub ncol: u32,
    pub row_stride: u32,
    pub col_stride: u32,
    _phantom: PhantomData<T>,
}

impl<T> MatrixLayout<T> {
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
        std::mem::swap(&mut self.nrow, &mut self.ncol);
        std::mem::swap(&mut self.row_stride, &mut self.col_stride);
        self
    }

    pub fn sliced(&self, rows: Range<u32>, cols: Range<u32>) -> Self {
        let rs = rows.start.min(self.nrow);
        let re = rows.end.max(rows.start).min(self.nrow);
        let cs = cols.start.min(self.ncol);
        let ce = cols.end.max(cols.start).min(self.ncol);

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
