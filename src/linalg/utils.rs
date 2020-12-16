use crate::linalg::Numeric;

/// Checks if given vector has only positive values, panics if not.
pub fn check_shape_positive(shape: &[usize]) {
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
pub fn check_shapes_the_same(shape1: &[usize], shape2: &[usize]) {
    if shape1 != shape2 {
        panic!("Arrays' shapes differ. Got: {:?} and {:?}", shape1, shape2);
    }
}

// Checks if arrays with given shapes can by multiplied, panics if not.
// TODO: add tests
pub fn check_shapes_matmul_arrays(arr1_shape: &[usize], arr2_shape: &[usize]) {
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
pub fn transpose_2d_matrix_slices<T: Numeric>(
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
}
