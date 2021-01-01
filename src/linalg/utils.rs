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

// Checks if arrays with given shapes can by multiplied, panics if not.
pub(crate) fn check_shapes_broadcast_matmul(arr1_shape: &[usize], arr2_shape: &[usize]) {
    let arr1_shape_len = arr1_shape.len();
    let arr2_shape_len = arr2_shape.len();
    check_shapes_broadcast(
        &arr1_shape[..arr1_shape_len - 2],
        &arr2_shape[..arr2_shape_len - 2],
    );
    if arr1_shape_len < 2
        || arr2_shape_len < 2
        || arr1_shape[arr1_shape_len - 1] != arr2_shape[arr2_shape_len - 2]
    {
        panic!(
            "Incompatible shapes for matrix product. Got: {:?} and {:?}",
            arr1_shape, arr2_shape
        );
    }
}

// Checks if two shapes are compatible in terms of array broadcasting, panics if not.
pub(crate) fn check_shapes_broadcast(shape1: &[usize], shape2: &[usize]) {
    let shape1_len = shape1.len();
    let shape2_len = shape2.len();

    let (smaller_shape, smaller_shape_len, bigger_shape, bigger_shape_len) =
        if shape1_len > shape2_len {
            (shape2, shape2_len, shape1, shape1_len)
        } else {
            (shape1, shape1_len, shape2, shape2_len)
        };

    let shapes_len_diff = bigger_shape_len - smaller_shape_len;
    for i in 0..smaller_shape_len {
        if smaller_shape[i] != bigger_shape[i + shapes_len_diff]
            && smaller_shape[i] != 1
            && bigger_shape[i + shapes_len_diff] != 1
        {
            panic!(
                "Given shapes aren't compatible for broadcast. Got: {:?} and {:?}",
                shape1, shape2
            )
        }
    }
}

// Returns shape of an array after applying element-wise operator on two arrays.
// Panics if shapes aren't compatible in terms of array broadcast.
pub(crate) fn get_shape_after_broadcast(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    check_shapes_broadcast(shape1, shape2);

    let shape1_len = shape1.len();
    let shape2_len = shape2.len();

    let (smaller_shape, smaller_shape_len, bigger_shape, bigger_shape_len) =
        if shape1_len > shape2_len {
            (shape2, shape2_len, shape1, shape1_len)
        } else {
            (shape1, shape1_len, shape2, shape2_len)
        };

    let mut new_shape = bigger_shape.to_vec();
    let shapes_lens_diff = bigger_shape_len - smaller_shape_len;
    for i in 0..smaller_shape_len {
        *new_shape.get_mut(i + shapes_lens_diff).unwrap() =
            smaller_shape[i].max(bigger_shape[i + shapes_lens_diff]);
    }
    new_shape
}

// Returns shape of an array after applying matrix product operator.
// Panics if shapes aren't compatible in terms of array broadcast or matrix product.
pub(crate) fn get_shape_after_broadcast_matmul(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    check_shapes_broadcast_matmul(shape1, shape2);
    let mut new_shape =
        get_shape_after_broadcast(&shape1[..shape1.len() - 2], &shape2[..shape2.len() - 2]);
    new_shape.push(shape1[shape1.len() - 2]);
    new_shape.push(*shape2.last().unwrap());
    new_shape
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

// Checks if given reduce axis is valid for given shape vector.
fn check_reduce_axis(shape: &[usize], axis: Option<usize>) {
    if let Some(axis_val) = axis {
        if axis_val >= shape.len() {
            panic!(
                "Invalid reduction dimension! Got shape: {:?} and dimension: {}.",
                shape, axis_val
            )
        }
    }
}

// Returns shape vector after applying reduce operator.
pub(crate) fn get_shape_after_reduce(
    shape: &[usize],
    axis: Option<usize>,
    keep_dims: bool,
) -> Vec<usize> {
    check_reduce_axis(shape, axis);
    if let Some(axis_val) = axis {
        let mut new_shape = shape.to_vec();
        if keep_dims {
            new_shape[axis_val] = 1;
        } else {
            new_shape.remove(axis_val);
        }
        new_shape
    } else if keep_dims {
        vec![1; shape.len()]
    } else {
        vec![1]
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

    #[test]
    fn test_get_shape_after_reduce() {
        assert_eq!(
            vec![2, 2],
            get_shape_after_reduce(&[2, 3, 2], Some(1), false)
        );
        assert_eq!(
            vec![2, 1, 2],
            get_shape_after_reduce(&[2, 3, 2], Some(1), true)
        );
        assert_eq!(
            vec![3, 2],
            get_shape_after_reduce(&[2, 3, 2], Some(0), false)
        );
        assert_eq!(
            vec![2, 3],
            get_shape_after_reduce(&[2, 3, 2], Some(2), false)
        );
        assert_eq!(vec![1], get_shape_after_reduce(&[2, 3, 2], None, false));
        assert_eq!(
            vec![1, 1, 1],
            get_shape_after_reduce(&[2, 3, 2], None, true)
        );
    }

    #[test]
    fn test_check_shapes_broadcast() {
        check_shapes_broadcast(&[2, 2, 3], &[2, 2, 3]);
        check_shapes_broadcast(&[1], &[2, 3, 2]);
        check_shapes_broadcast(&[1, 1, 1], &[2, 3, 2]);
        check_shapes_broadcast(&[3, 1, 1], &[1, 3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_check_shapes_broadcast_wrong_last_dim() {
        check_shapes_broadcast(&[2, 2, 2], &[2, 2, 3]);
    }

    #[test]
    fn test_check_shapes_broadcast_matmul() {
        check_shapes_broadcast_matmul(&[2, 2, 3], &[2, 3, 2]);
        check_shapes_broadcast_matmul(&[1, 2, 3], &[2, 3, 2]);
        check_shapes_broadcast_matmul(&[2, 2, 3], &[1, 3, 2]);
        check_shapes_broadcast_matmul(&[2, 3], &[7, 3, 2]);
        check_shapes_broadcast_matmul(&[2, 2, 3], &[3, 2]);
        check_shapes_broadcast_matmul(&[3, 2, 2, 3], &[3, 1, 3, 2]);
        check_shapes_broadcast_matmul(&[1, 5, 2, 3], &[3, 1, 3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_check_shapes_broadcast_matmul_wrong_last_dims() {
        check_shapes_broadcast_matmul(&[2, 2, 2], &[2, 3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_check_shapes_broadcast_matmul_wrong_broadcast() {
        check_shapes_broadcast_matmul(&[2, 2, 3], &[3, 3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_check_shapes_broadcast_matmul_too_short_first() {
        check_shapes_broadcast_matmul(&[3], &[3, 2]);
    }

    #[test]
    #[should_panic]
    fn test_check_shapes_broadcast_matmul_too_short_second() {
        check_shapes_broadcast_matmul(&[3, 2], &[2]);
    }

    #[test]
    fn test_get_shape_after_broadcast() {
        assert_eq!(
            get_shape_after_broadcast(&[3, 5, 2], &[3, 5, 2]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[3, 5, 2], &[5, 2]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[5, 2], &[3, 5, 2]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[3, 5, 2], &[1, 5, 2]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[3, 1, 2], &[3, 5, 2]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[3, 5, 2], &[3, 5, 1]),
            vec![3, 5, 2]
        );
        assert_eq!(
            get_shape_after_broadcast(&[3, 5, 1], &[3, 5, 1]),
            vec![3, 5, 1]
        );
        assert_eq!(get_shape_after_broadcast(&[3, 1], &[1, 3]), vec![3, 3]);
    }

    #[test]
    fn test_get_shape_after_broadcast_matmul() {
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 5, 2, 3, 2], &[3, 5, 2, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 5, 2, 3, 2], &[5, 2, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[5, 2, 3, 2], &[3, 5, 2, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 5, 2, 3, 2], &[1, 5, 2, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 1, 2, 3, 2], &[3, 5, 2, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 5, 2, 3, 2], &[3, 5, 1, 2, 3]),
            vec![3, 5, 2, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 5, 1, 3, 2], &[3, 5, 1, 2, 3]),
            vec![3, 5, 1, 3, 3]
        );
        assert_eq!(
            get_shape_after_broadcast_matmul(&[3, 1, 3, 2], &[1, 3, 2, 3]),
            vec![3, 3, 3, 3]
        );
    }

    #[test]
    #[should_panic]
    fn test_get_shape_after_broadcast_wrong_matrix_shape() {
        get_shape_after_broadcast_matmul(&[3, 5, 2, 3, 2], &[3, 5, 2, 3, 3]);
    }
}
