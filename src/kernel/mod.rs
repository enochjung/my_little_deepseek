pub(crate) mod x86_64;

#[cfg(test)]
mod tests {
    use super::x86_64::{mul, rms};

    #[test]
    fn case01_rms_n5() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let actual = unsafe { rms(x.as_ptr(), x.len()) };
        let expected = (11.0f32).sqrt();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn case02_rms_n16() {
        let x: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let actual = unsafe { rms(x.as_ptr(), x.len()) };
        let expected = (93.5f32).sqrt();
        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn case03_mul_n5() {
        let x = [2.0f32, 4.0, 6.0, 8.0, 10.0];
        let mut y = [1.0f32, 2.0, 3.0, 4.0, 5.0];

        unsafe { mul(y.as_mut_ptr(), x.as_ptr(), 0.5, x.len()) };

        let expected = [1.0f32, 4.0, 9.0, 16.0, 25.0];
        for i in 0..y.len() {
            assert!((y[i] - expected[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn case04_mul_n16() {
        let x: [f32; 16] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let mut y = [1.0f32; 16];

        unsafe { mul(y.as_mut_ptr(), x.as_ptr(), 2.0, x.len()) };

        let expected: [f32; 16] = [
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0,
        ];
        for i in 0..y.len() {
            assert!((y[i] - expected[i]).abs() < 1e-6);
        }
    }
}
