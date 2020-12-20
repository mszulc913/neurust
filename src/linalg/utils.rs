use crate::linalg::Numeric;
use crate::Array;

/// Checks if two floating point numbers are relatively equal.
///
/// See https://floating-point-gui.de/errors/comparison/
///
/// - `a` - First number to be compared.
/// - `b` - Second number to be compared.'
/// - `epsilon` - Error marigin, very small number.
///
/// # Examples
/// ```
/// use neurust::linalg::utils::are_numbers_near_equal;
///
/// assert!(!are_numbers_near_equal(1., 1.5, 1e-7));
/// assert!(are_numbers_near_equal(1., 1.000000001, 1e-7));
/// ```
pub fn are_numbers_near_equal<T: Numeric>(a: T, b: T, epsilon: T) -> bool {
    let abs_a = a.abs();
    let abs_b = b.abs();
    let diff = (abs_a - abs_b).abs();
    if a == b {
        true
    } else if a == T::zero() || b == T::zero() || abs_a + abs_b < T::min_positive_value() {
        diff < epsilon * T::min_positive_value()
    } else {
        diff / T::min(abs_a + abs_b, T::max_value()) < epsilon
    }
}

/// Checks if two floating point arrays are relatively equal.
///
/// This function can be used via `assert_arrays_rel_eq` macro.
/// See https://floating-point-gui.de/errors/comparison/
///
/// - `a` - First array to be compared.
/// - `b` - Second array to be compared.'
/// - `epsilon` - Error marigin, very small number.
///
/// # Examples
/// ```
/// use neurust::linalg::utils::are_arrays_near_equal;
/// use neurust::prelude::*;
///
/// assert!(!are_arrays_near_equal(
///     &Array::from_vec(vec![1., 2., 3., 4.], vec![2, 2]),
///     &Array::from_vec(vec![1., 2.001, 3., 4.], vec![2, 2]),
///     1e-7
/// ));
/// assert!(are_arrays_near_equal(
///     &Array::from_vec(vec![1., 2., 3., 4.], vec![2, 2]),
///     &Array::from_vec(vec![1., 2.0000000001, 3., 4.], vec![2, 2]),
///     1e-7
/// ));
/// ```
pub fn are_arrays_near_equal<T: Numeric>(a: &Array<T>, b: &Array<T>, epsilon: T) -> bool {
    if a.shape != b.shape {
        false
    } else {
        a.data
            .iter()
            .zip(b.data.iter())
            .all(|x| are_numbers_near_equal(*x.0, *x.1, epsilon))
    }
}

/// Checks if two floating point arrays are relatively equal.
///
/// See https://floating-point-gui.de/errors/comparison/
///
/// # Examples
/// ```
/// # #[macro_use] extern crate neurust;
/// use neurust::prelude::*;
/// use neurust::linalg::utils::are_arrays_near_equal;
/// # fn main() {
/// let a = Array::from_vec(vec![0., 0., 1., 1.], vec![2, 2]);
/// let b = Array::from_vec(vec![0., 0., 1.000000000001, 1.], vec![2, 2]);
///
/// assert_arrays_rel_eq!(a, b, 1e-7);
/// # }
/// ```
#[macro_export]
macro_rules! assert_arrays_rel_eq {
    ($left:expr, $right:expr, $epsilon: expr) => {{
        if !are_arrays_near_equal(&$left, &$right, $epsilon) {
            panic!(
                r#"assertion failed: `(left == right)`
 left: `{:?}`,
 right: `{:?}`"#,
                $left, $right
            )
        }
    }};
}

/// Checks if given vector has only positive values, panics if not.
pub(crate) fn check_shape_positive(shape: &[usize]) {
    for dim in shape.iter() {
        if *dim == 0 {
            panic!(
                "Shape should only contain positive numbers. Got: {:?}",
                shape
            )
        }
    }
}

/// Checks if two shapes are equal, panics if not.
pub(crate) fn check_shapes_the_same(shape1: &[usize], shape2: &[usize]) {
    if shape1 != shape2 {
        panic!("Arrays' shapes differ. Got: {:?} and {:?}", shape1, shape2);
    }
}

// Checks if arrays with given shapes can by multiplied, panics if not.
// TODO: add tests
pub(crate) fn check_shapes_matmul_arrays(arr1_shape: &[usize], arr2_shape: &[usize]) {
    let mut are_shapes_ok = true;
    let arr1_shape_len = arr1_shape.len();
    let arr2_shape_len = arr2_shape.len();
    if arr1_shape_len != arr2_shape_len
        || arr2_shape_len < 2 && arr1_shape[arr1_shape_len - 1] != arr2_shape[arr1_shape_len - 2]
    {
        are_shapes_ok = false;
    } else {
        for i in 0..(arr1_shape_len - 2) {
            if arr1_shape[i] != arr2_shape[i] {
                are_shapes_ok = false;
                break;
            }
        }
    }
    if !are_shapes_ok {
        panic!(
            "Incompatible shapes. Got: {:?} and {:?}",
            arr1_shape, arr2_shape
        );
    }
}

/// Transposes matrix to a given location.
pub(crate) fn transpose_2d_matrix_slices<T: Numeric>(
    data: &[T],
    n_rows: usize,
    n_cols: usize,
    output_buffer: &mut [T],
) {
    if output_buffer.len() != n_cols * n_rows {
        panic!(
            "Output buffer has wrong length. Got: {}, expected: {}",
            output_buffer.len(),
            n_cols * n_rows
        )
    }
    if data.len() != n_rows * n_cols {
        panic!(
            "Input data slice has wrong length. Got: {}, expected: {}",
            data.len(),
            n_rows * n_cols
        )
    }

    for i in 0..n_rows {
        for j in 0..n_cols {
            output_buffer[j * n_rows + i] = data[i * n_cols + j];
        }
    }
}

#[cfg(test)]
mod tests {
    pub use super::*;

    #[test]
    fn test_check_shape_positive() {
        let shape = vec![1, 2, 3];

        check_shape_positive(&shape);
    }

    #[test]
    fn test_check_shape_positive_empty() {
        let shape = vec![];

        check_shape_positive(&shape);
    }

    #[should_panic]
    #[test]
    fn test_check_shape_positive_panic() {
        let shape = vec![1, 0, 3];

        check_shape_positive(&shape);
    }

    #[test]
    fn test_transpose_2d_matrix_slices() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff = vec![0.0; 6];

        transpose_2d_matrix_slices(&a, 2, 3, &mut output_buff);

        assert_eq!(output_buff, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_transpose_2d_matrix_slices_output_wrong_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff = vec![0.0; 5];

        transpose_2d_matrix_slices(&a, 2, 3, &mut output_buff);
    }

    #[test]
    #[should_panic]
    fn test_transpose_2d_matrix_slices_input_wrong_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output_buff = vec![0.0; 6];

        transpose_2d_matrix_slices(&a, 2, 4, &mut output_buff);
    }

    #[test]
    fn test_are_numbers_near_equal() {
        assert!(are_numbers_near_equal(1., 1., 1e-7));
        assert!(are_numbers_near_equal(0., 0., 1e-7));
        assert!(!are_numbers_near_equal(0.001, 0., 1e-7));
        assert!(are_numbers_near_equal(10.0000001, 10.000000000001, 1e-7));
    }

    #[test]
    #[should_panic]
    fn test_are_arrays_rel_eq_panics() {
        let a = Array::from_vec(vec![0., 0., 1., 1.], vec![2, 2]);
        let b = Array::from_vec(vec![0., 0.000001, 1., 1.], vec![2, 2]);

        assert_arrays_rel_eq!(a, b, 1e-7)
    }
}
