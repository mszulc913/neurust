mod arithmetic;
pub mod math;

use crate::graph::{GraphOp, Placeholder, Variable};
use crate::linalg::{Array, Numeric};

use crate::graph::arithmetic::MatMulOp;
use std::collections::HashMap;
use std::rc::Rc;

/// Represents a node of a computational graph that operates on n-dimensional arrays.
///
/// Tensor objects can be evaluated (*forward pass*) and differentiated
/// (*backward pass*). In current version of the library all tensors should be
/// of the same type `T` in order to be placed in the same computational graph.
///
/// * `op` - Shared reference to a computational graph node.
pub struct Tensor<T: Numeric> {
    op: Rc<(dyn GraphOp<T>)>,
}

impl<T: Numeric> Tensor<T> {
    /// Evaluates a tensor computing its value.
    ///
    /// * `feed_dict` - Dictionary with values for *placeholder* tensors current tensor
    /// is dependant of.
    ///
    /// **Panics** if `feed_dict` does not contain required data or if shapes
    /// of tensors in a graph are invalid.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let a = get_variable(Array::from_vec(vec![0., 1., 2., 3.], vec![2, 2]));
    /// let b = get_variable(Array::from_vec(vec![4., 5., 6., 7.], vec![2, 2]));
    /// let add = &a + &b; // `add` is a tensor!
    ///
    /// assert_eq!(
    ///     add.eval(None),
    ///     Array::from_vec(vec![4., 6., 8., 10.], vec![2, 2])
    /// )
    /// ```
    pub fn eval(&self, feed_dict: Option<HashMap<String, &Array<T>>>) -> Array<T> {
        self.op.eval(feed_dict)
    }

    /// Computes gradient of a tensor with respect to `y` tensor.
    ///
    /// Gradient is computed via reverse accumulation automatic differentiation.
    /// If `y` is not connected to a tensor in any way, then `None` is returned.
    ///
    /// * `feed_dict` - Dictionary with values for *placeholder* tensors current tensor
    /// is dependant of.
    ///
    /// **Panics** if `feed_dict` does not contain required data or if shapes
    /// of tensors in a graph are invalid.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let a = get_variable(Array::from_vec(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]));
    /// let b = get_variable(Array::from_vec(vec![4., 5., 6.], vec![3, 1]));
    /// let mul = a.matmul(&b); // matrix product of a and b
    ///
    /// assert_eq!(
    ///     mul.grad(&b, None).unwrap(),
    ///     Array::from_vec(vec![3., 5., 7.], vec![3, 1])
    /// )
    /// ```
    pub fn grad(
        &self,
        y: &Tensor<T>,
        feed_dict: Option<HashMap<String, &Array<T>>>,
    ) -> Option<Array<T>> {
        self.op.grad(y.op.as_ref(), feed_dict)
    }

    /// Creates a tensor that evaluates to matrix product of two tensors.
    ///
    /// Tensors can be multiplied only if:
    /// * their shapes have the same length
    /// * they are at least 2 dimensional
    /// * last two dimensions of both matrices are valid in terms of matrix multiplication
    /// and rest of the dimensions are pair-wise equal, that is their shapes are in the following
    /// form: `[a, b, ..., d, e, f]` x `[a, b, ..., d, f, g]`.
    /// The resulting array will have the shape of `[a, b, ..., d, e, g]`.
    /// Tensors are multiplied in a such way that pairs of sub-arrays of shape
    /// `[e, f]` and `[f, g]` are multiplied in a standard way (standard matrix product)
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let a = get_variable(Array::from_vec(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]));
    /// let b = get_variable(Array::from_vec(vec![4., 5., 6.], vec![3, 1]));
    /// let mul = a.matmul(&b);
    ///
    /// assert_eq!(
    ///     mul.eval(None),
    ///     Array::from_vec(vec![17., 62.], vec![2, 1])
    /// )
    /// ```
    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T> {
        Tensor {
            op: Rc::new(MatMulOp::new(Rc::clone(&self.op), Rc::clone(&other.op))),
        }
    }
}

/// Returns new *variable* Tensor.
///
/// Variables represent persistent memory that changes over time (for example in a process
/// of *learning*)
///
/// * `init_value` - Value a tensor should be initialized with.
///
/// # Examples
/// ```
/// use neurust::prelude::*;
///
/// let a = get_variable(Array::from_vec(vec![0., 1., 2.], vec![1, 3]));
///
/// assert_eq!(
///     a.eval(None),
///     Array::from_vec(vec![0., 1., 2.], vec![1, 3])
/// )
/// ```
pub fn get_variable<T: Numeric>(init_value: Array<T>) -> Tensor<T> {
    Tensor {
        op: Rc::new(Variable::new(init_value)),
    }
}

/// Returns new *placeholder* Tensor.
///
/// Placeholders are tensors without any data at initialization step.
/// Instead, the data should be provided later in `feed_dict`.
///
/// * `id` - Unique name of the placeholder.
///
/// # Examples
/// ```
/// use neurust::prelude::*;
/// use std::collections::HashMap;
///
/// let a = get_placeholder("some_value".to_string());
/// let a_value = Array::from_vec(vec![0., 1., 2.], vec![1, 3]);
/// let mut feed_dict = HashMap::new();
/// feed_dict.insert(
///    "some_value".to_owned(),
///    &a_value
/// );
///
/// assert_eq!(
///     a.eval(Some(feed_dict)),
///     Array::from_vec(vec![0., 1., 2.], vec![1, 3])
/// )
/// ```
pub fn get_placeholder<T: Numeric>(id: String) -> Tensor<T> {
    Tensor {
        op: Rc::new(Placeholder::new(id)),
    }
}
