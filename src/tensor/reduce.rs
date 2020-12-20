use crate::graph::reduce::{ReduceMeanOp, ReduceSumOp};
use crate::linalg::Numeric;
use crate::Tensor;
use std::rc::Rc;

/// Computes a sum of elements of a tensor across dimensions.
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
/// use neurust::{Tensor, Array, reduce_sum};
///
/// let var = Tensor::new_variable(Array::from_vec(
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
/// ));
///
/// assert_eq!(
///     reduce_sum(&var, None, false).eval(None),
///     Array::new(66., vec![1])
/// );
/// assert_eq!(
///     reduce_sum(&var, Some(1), false).eval(None),
///     Array::from_vec(
///         vec![
///             6., 9.,
///             24., 27.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_sum(&var, Some(1), true).eval(None),
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
pub fn reduce_sum<T: Numeric>(
    tensor: &Tensor<T>,
    axis: Option<usize>,
    keep_dims: bool,
) -> Tensor<T> {
    Tensor::new(Rc::new(ReduceSumOp::new(
        Rc::clone(&tensor.op),
        axis,
        keep_dims,
    )))
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
/// use neurust::{Array, reduce_mean, Tensor};
///
/// let var = Tensor::new_variable(Array::from_vec(
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
/// ));
///
/// assert_eq!(
///     reduce_mean(&var, None, false).eval(None),
///     Array::new(5.5, vec![1])
/// );
/// assert_eq!(
///     reduce_mean(&var, Some(1), false).eval(None),
///     Array::from_vec(
///         vec![
///             2., 3.,
///             8., 9.
///         ],
///         vec![2, 2]
///     )
/// );
/// assert_eq!(
///     reduce_mean(&var, Some(1), true).eval(None),
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
pub fn reduce_mean<T: Numeric>(
    tensor: &Tensor<T>,
    axis: Option<usize>,
    keep_dims: bool,
) -> Tensor<T> {
    Tensor::new(Rc::new(ReduceMeanOp::new(
        Rc::clone(&tensor.op),
        axis,
        keep_dims,
    )))
}
