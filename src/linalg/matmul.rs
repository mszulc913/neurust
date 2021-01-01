extern crate cblas_sys as ffi;
extern crate openblas_src;

use crate::linalg::Numeric;
use core::any::TypeId;

/// Matrix multiplication of two data slices. Result is stored
/// in a given buffer slice.
///
/// If `T` is `f32` or `f64`, then *BLAS* is used.
pub(crate) fn matmul_2d_matrix_slices<T: Numeric>(
    data1: &[T],
    n_rows1: usize,
    n_cols1: usize,
    data2: &[T],
    n_rows2: usize,
    n_cols2: usize,
    output_buffer: &mut [T],
) {
    check_matrix_product_shapes(
        data1,
        n_rows1,
        n_cols1,
        data2,
        n_rows2,
        n_cols2,
        output_buffer,
    );
    let (m, n, k) = (n_rows1 as i32, n_cols2 as i32, n_cols1 as i32);
    let dt = TypeId::of::<T>();
    if dt == TypeId::of::<f32>() {
        unsafe {
            ffi::cblas_sgemm(
                ffi::CblasRowMajor,
                ffi::CblasNoTrans,
                ffi::CblasNoTrans,
                m,
                n,
                k,
                1.0,
                data1.as_ptr() as *const f32,
                k,
                data2.as_ptr() as *const f32,
                n,
                1.0,
                output_buffer.as_mut_ptr() as *mut f32,
                n,
            );
        }
    } else if dt == TypeId::of::<f64>() {
        unsafe {
            ffi::cblas_dgemm(
                ffi::CblasRowMajor,
                ffi::CblasNoTrans,
                ffi::CblasNoTrans,
                m,
                n,
                k,
                1.0,
                data1.as_ptr() as *const f64,
                k,
                data2.as_ptr() as *const f64,
                n,
                1.0,
                output_buffer.as_mut_ptr() as *mut f64,
                n,
            );
        }
    } else {
        general_matmul_2d_matrix_slices(data1, n_rows1, n_cols1, data2, n_cols2, output_buffer)
    }
}

fn general_matmul_2d_matrix_slices<T: Numeric>(
    data1: &[T],
    n_rows1: usize,
    n_cols1: usize,
    data2: &[T],
    n_cols2: usize,
    output_buffer: &mut [T],
) {
    for i in 0..n_rows1 {
        for j in 0..n_cols2 {
            let mut sum = T::zero();
            for k in 0..n_cols1 {
                sum = sum + data1[i * n_cols1 + k] * data2[k * n_cols2 + j];
            }
            output_buffer[i * n_cols2 + j] = sum;
        }
    }
}

fn check_matrix_product_shapes<T: Numeric>(
    data1: &[T],
    n_rows1: usize,
    n_cols1: usize,
    data2: &[T],
    n_rows2: usize,
    n_cols2: usize,
    output_buffer: &[T],
) {
    if n_cols1 != n_rows2 {
        panic!(
            "Inner dimensions of the matrices doesn't match. Got shapes: [{}, {}] and [{}, {}].",
            n_rows1, n_cols1, n_rows2, n_cols2
        )
    }
    if output_buffer.len() != n_cols2 * n_rows1 {
        panic!(
            "Output buffer has wrong length. Got: {}, expected: {}",
            output_buffer.len(),
            n_cols2 * n_rows1
        )
    }
    if data1.len() != n_rows1 * n_cols1 {
        panic!(
            "First data slice has wrong length. Got: {}, expected: {}",
            data1.len(),
            n_rows1 * n_cols1
        )
    }
    if data2.len() != n_rows2 * n_cols2 {
        panic!(
            "Second data slice has wrong length. Got: {}, expected: {}",
            data2.len(),
            n_rows2 * n_cols2
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d_matrix_slices_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff: Vec<f32> = vec![0.0; 4];

        matmul_2d_matrix_slices(&a, 2, 3, &b, 3, 2, &mut output_buff);

        assert_eq!(output_buff, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_2d_matrix_slices_f64() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff: Vec<f64> = vec![0.0; 4];

        matmul_2d_matrix_slices(&a, 2, 3, &b, 3, 2, &mut output_buff);

        assert_eq!(output_buff, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_general_matmul() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff: Vec<f64> = vec![0.0; 4];

        matmul_2d_matrix_slices(&a, 2, 3, &b, 3, 2, &mut output_buff);

        assert_eq!(output_buff, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[should_panic]
    #[test]
    fn test_matmul_2d_matrix_slices_wrong_shapes() {
        let a = vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut output_buff = vec![0.0; 4];

        matmul_2d_matrix_slices(&a, 2, 4, &b, 3, 2, &mut output_buff);
    }

    #[should_panic]
    #[test]
    fn test_matmul_2d_matrix_slices_output_wrong_length() {
        let a = vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut output_buff = vec![0.0; 5];

        matmul_2d_matrix_slices(&a, 2, 3, &b, 3, 2, &mut output_buff);
    }

    #[should_panic]
    #[test]
    fn test_matmul_2d_matrix_slices_input_wrong_length() {
        let a = vec![1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut output_buff = vec![0.0; 4];

        matmul_2d_matrix_slices(&a, 2, 4, &b, 3, 2, &mut output_buff);
    }
}
