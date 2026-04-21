use super::{DataType, F32, HostMemory, OpMV};

pub struct OpMM<D: DataType, S>(core::marker::PhantomData<(D, S)>);

impl OpMM<F32, HostMemory> {
    /// Computes `C = C * A + B` in-place.
    ///
    /// Shapes:
    /// - `C`: `(m, n)` (input and output)
    /// - `A`: `(n, n)`
    /// - `B`: `(1, n)` and broadcasted row-wise over `C`
    ///
    /// Parameters:
    /// - `c`: Mutable f32 slice for `C`.
    /// - `a`: Flattened row-major buffer for `A`.
    /// - `b`: Flattened row-major buffer for `B`.
    /// - `m`: Number of rows in `C`.
    /// - `n`: Number of columns in `C`.
    pub fn muladd_mn_nn_1n(c: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) -> () {
        debug_assert_eq!(c.len(), m * n);
        debug_assert_eq!(a.len(), n * n);
        debug_assert_eq!(b.len(), n);

        for row in c.chunks_exact_mut(n) {
            OpMV::<F32, HostMemory>::muladd_1n_nn_1n(row, a, b, n);
        }
    }
}
