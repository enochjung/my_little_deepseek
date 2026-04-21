use super::{DataType, F32, HostMemory};

pub struct OpMV<D: DataType, S>(core::marker::PhantomData<(D, S)>);

impl OpMV<F32, HostMemory> {
    /// Computes `C = C * A + B` in-place for a single row vector.
    ///
    /// Shapes:
    /// - `C`: `(1, n)` (input and output)
    /// - `A`: `(n, n)`
    /// - `B`: `(1, n)`
    pub fn muladd_1n_nn_1n(c: &mut [f32], a: &[f32], b: &[f32], n: usize) -> () {
        debug_assert_eq!(c.len(), n);
        debug_assert_eq!(a.len(), n * n);
        debug_assert_eq!(b.len(), n);

        let input = c.to_vec();

        for out_col in 0..n {
            let mut value = b[out_col];
            for in_col in 0..n {
                value += input[in_col] * a[in_col * n + out_col];
            }
            c[out_col] = value;
        }
    }
}
