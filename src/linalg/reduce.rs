use crate::linalg::Numeric;
use crate::Array;
use num::cast;

fn check_reduce_axis<T: Numeric>(array: &Array<T>, axis: Option<usize>) {
    if let Some(axis_val) = axis {
        let shape = array.get_shape();
        if axis_val >= shape.len() {
            panic!(
                "Invalid reduction dimension! Got shape: {:?} and dimension: {}.",
                shape, axis_val
            )
        }
    }
}

fn get_shape_after_reduce<T: Numeric>(
    array: &Array<T>,
    axis: Option<usize>,
    keep_dims: bool,
) -> Vec<usize> {
    check_reduce_axis(array, axis);
    if let Some(axis_val) = axis {
        let mut shape = array.get_shape();
        if keep_dims {
            shape[axis_val] = 1;
        } else {
            shape.remove(axis_val);
        }
        shape
    } else if keep_dims {
        vec![1; array.get_shape().len()]
    } else {
        vec![1]
    }
}

/// Reduces given dimension to a single value by applying
/// *reducer* function to the data.
///
/// If `None` is passed, all dimensions are reduced.
///
/// * `axis` - The dimension to reduce.
/// * `reducer` - Function to be applied.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than or equal to the length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce(&arr, |x, y| x + y, None, false),
///     Array::new(66., vec![1])
/// );
/// assert_eq!(
///     reduce(&arr, |x, y| x + y, Some(1), false),
///     Array::from_vec(
///         vec![
///             6., 9.,
///             24., 27.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce(&arr, |x, y| x + y, Some(1), true),
///     Array::from_vec(
///         vec![
///             6., 9.,
///
///             24., 27.
///         ],
///         vec![2, 1, 2]
///     )
/// );
/// ```
pub fn reduce<T: Numeric>(
    array: &Array<T>,
    reducer: fn(T, T) -> T,
    axis: Option<usize>,
    keep_dims: bool,
) -> Array<T> {
    let new_shape = get_shape_after_reduce(array, axis, keep_dims);
    let mut new_data = vec![T::zero(); new_shape.iter().product()];

    if let Some(axis_val) = axis {
        let axis_len: usize = array.shape[axis_val + 1..].iter().product();
        let single_slide: usize = array.shape[axis_val..].iter().product();
        let mut processed_elems = 0;
        let mut total_slide = 0;
        let mut current_row = 0;
        let dim_len = array.shape[axis_val];
        for output_elem in new_data.iter_mut() {
            *output_elem = array.data[total_slide + current_row];
            processed_elems += 1;
            for j in 1..dim_len {
                processed_elems += 1;
                *output_elem = reducer(
                    *output_elem,
                    array.data[total_slide + axis_len * j + current_row],
                );
            }
            current_row += 1;
            if processed_elems % single_slide == 0 {
                total_slide += single_slide;
                current_row = 0;
            }
        }
    } else {
        new_data[0] = array.data.iter().fold(T::zero(), |acc, x| reducer(acc, *x));
    }

    Array {
        data: new_data,
        shape: new_shape,
    }
}

/// Computes a sum of elements of an array across dimensions.
///
/// If `None` is passed, sum of all array elements is computed.
///
/// * `axis` - The dimension to reduce.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than equal to length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce_sum};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce_sum(&arr, None, false),
///     Array::new(66., vec![1])
/// );
/// assert_eq!(
///     reduce_sum(&arr, Some(1), false),
///     Array::from_vec(
///         vec![
///             6., 9.,
///             24., 27.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_sum(&arr, Some(1), true),
///     Array::from_vec(
///         vec![
///             6., 9.,
///
///             24., 27.
///         ],
///         vec![2, 1, 2]
///     )
/// );
/// ```
pub fn reduce_sum<T: Numeric>(array: &Array<T>, axis: Option<usize>, keep_dims: bool) -> Array<T> {
    reduce(array, |x, y| x + y, axis, keep_dims)
}

/// Computes a product of elements of an array across dimensions.
///
/// If `None` is passed, product of all array elements is computed.
///
/// * `axis` - The dimension to reduce.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than equal to length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce_prod};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce_prod(&arr, None, false),
///     Array::new(0., vec![1])
/// );
/// assert_eq!(
///     reduce_prod(&arr, Some(1), false),
///     Array::from_vec(
///         vec![
///             0., 15.,
///             480., 693.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_prod(&arr, Some(1), true),
///     Array::from_vec(
///         vec![
///             0., 15.,
///
///             480., 693.
///         ],
///         vec![2, 1, 2]
///     )
/// );
/// ```
pub fn reduce_prod<T: Numeric>(array: &Array<T>, axis: Option<usize>, keep_dims: bool) -> Array<T> {
    reduce(array, |x, y| x * y, axis, keep_dims)
}

/// Computes a maximum of elements of an array across dimensions.
///
/// If `None` is passed, maximum of all array elements is computed.
///
/// * `axis` - The dimension to reduce.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than equal to length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce_max};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce_max(&arr, None, false),
///     Array::new(11., vec![1])
/// );
/// assert_eq!(
///     reduce_max(&arr, Some(1), false),
///     Array::from_vec(
///         vec![
///             4., 5.,
///             10., 11.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_max(&arr, Some(1), true),
///     Array::from_vec(
///         vec![
///             4., 5.,
///
///             10., 11.
///         ],
///         vec![2, 1, 2]
///     )
/// );
///
/// ```
pub fn reduce_max<T: Numeric>(array: &Array<T>, axis: Option<usize>, keep_dims: bool) -> Array<T> {
    reduce(array, |x, y| x.max(y), axis, keep_dims)
}

/// Computes a minimum of elements of an array across dimensions.
///
/// If `None` is passed, sum of all array elements is computed.
///
/// * `axis` - The dimension to reduce.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than equal to length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce_min};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce_min(&arr, None, false),
///     Array::new(0., vec![1])
/// );
/// assert_eq!(
///     reduce_min(&arr, Some(1), false),
///     Array::from_vec(
///         vec![
///             0., 1.,
///             6., 7.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_min(&arr, Some(1), true),
///     Array::from_vec(
///         vec![
///             0., 1.,
///
///             6., 7.
///         ],
///         vec![2, 1, 2]
///     )
/// );
///
/// ```
pub fn reduce_min<T: Numeric>(array: &Array<T>, axis: Option<usize>, keep_dims: bool) -> Array<T> {
    reduce(array, |x, y| x.min(y), axis, keep_dims)
}

/// Computes a mean of elements of an array across dimensions.
///
/// If `None` is passed, mean of all array elements is computed.
///
/// * `axis` - The dimension to reduce.
/// * `keep_dims` - If true, preserves reduced dimensions with length 1.
///
/// **Panics** if `axis` is more than equal to length of array's shape vector.
///
/// # Examples
/// ```
/// use neurust::linalg::{Array, reduce_mean};
///
/// let arr = Array::from_vec(
///     vec![
///         0., 1.,
///         2., 3.,
///         4., 5.,
///
///         6., 7.,
///         8., 9.,
///         10., 11.
///     ],
///     vec![2, 3, 2]
/// );
///
/// assert_eq!(
///     reduce_mean(&arr, None, false),
///     Array::new(5.5, vec![1])
/// );
/// assert_eq!(
///     reduce_mean(&arr, Some(1), false),
///     Array::from_vec(
///         vec![
///             2., 3.,
///             8., 9.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_mean(&arr, Some(1), true),
///     Array::from_vec(
///         vec![
///             2., 3.,
///
///             8., 9.
///         ],
///         vec![2, 1, 2]
///     )
/// );
///
/// ```
pub fn reduce_mean<T: Numeric>(array: &Array<T>, axis: Option<usize>, keep_dims: bool) -> Array<T> {
    let mut sum = reduce_sum(array, axis, keep_dims);
    let count = if let Some(axis_val) = axis {
        array.shape[axis_val]
    } else {
        array.data.len()
    };
    sum.div_assign_scalar(cast::<_, T>(count).unwrap());
    sum
}
