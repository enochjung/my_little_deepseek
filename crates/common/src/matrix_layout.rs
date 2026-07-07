use core::ops::{Bound, RangeBounds};

pub struct MatrixLayout {
    pub srow: u32,
    pub scol: u32,
    pub nrow: u32,
    pub ncol: u32,
    pub is_trans: bool,
}

impl MatrixLayout {
    pub fn new(srow: u32, scol: u32, nrow: u32, ncol: u32, is_trans: bool) -> Self {
        Self {
            srow,
            scol,
            nrow,
            ncol,
            is_trans,
        }
    }

    pub fn shape(&self) -> (u32, u32) {
        match self.is_trans {
            false => (self.nrow, self.ncol),
            true => (self.ncol, self.nrow),
        }
    }

    pub fn sliced(&self, rows: impl RangeBounds<u32>, cols: impl RangeBounds<u32>) -> Self {
        let (rse, cse) = match self.is_trans {
            false => {
                let rse = normalize(rows, self.nrow);
                let cse = normalize(cols, self.ncol);
                (rse, cse)
            }
            true => {
                let rse = normalize(cols, self.nrow);
                let cse = normalize(rows, self.ncol);
                (rse, cse)
            }
        };

        Self {
            srow: self.srow + rse.0,
            scol: self.scol + cse.0,
            nrow: rse.1 - rse.0,
            ncol: cse.1 - cse.0,
            is_trans: self.is_trans,
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
