mod arithmetic;
pub mod math;
mod reduce;

use crate::graph::{GraphOp, Placeholder, Variable};
use crate::linalg::{Array, Numeric};
pub use reduce::{reduce_mean, reduce_sum};

use crate::graph::arithmetic::MatMulOp;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Represents a node of a computational graph that operates on n-dimensional arrays.
///
/// Tensor objects can be evaluated (*forward pass*) and differentiated
/// (*backward pass*). In current version of the library all tensors should be
/// of the same type `T` in order to be placed in the same computational graph.
///
/// Computational graphs are being defined by applying functions and overloaded
/// Rust operators to tensors. Note that the library will panic if shapes of
/// used tensors won't be valid in a context of a specific operation.
///
/// * `op` - Shared reference to a computational graph node.
/// * `variable_data` - Shared reference to stored variable operator's data.
/// This is `None` for tensors with `op` field other than `Variable`.
pub struct Tensor<T: Numeric> {
    op: Rc<(dyn GraphOp<T>)>,
    variable_data: Option<Rc<RefCell<Array<T>>>>,
}

impl<T: Numeric> Tensor<T> {
    fn new(op: Rc<(dyn GraphOp<T>)>) -> Tensor<T> {
        Tensor {
            op,
            variable_data: None,
        }
    }

    /// Returns the shape of a tensor (shape of the output array after evaluation).
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let a = Tensor::new_variable(Array::from_vec(vec![0., 1., 2.], vec![1, 3]));
    ///
    /// assert_eq!(a.shape(), vec![1, 3]);
    /// ```
    pub fn shape(&self) -> Vec<usize> {
        self.op.shape()
    }

    /// Returns new *placeholder* Tensor.
    ///
    /// Placeholders are tensors without any data at the initialization step.
    /// Instead, the data should provided later, when the tensor is coputed.
    ///
    /// * `id` - Unique name of the placeholder.
    /// * `shape` - Shape vector.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    /// use std::collections::HashMap;
    ///
    /// let a = Tensor::new_placeholder("some_value".to_string(), vec![1, 3]);
    /// let a_value = Array::from_vec(vec![0., 1., 2.], vec![1, 3]);
    /// let mut feed_dict = HashMap::new();
    /// feed_dict.insert(
    ///    "some_value".to_owned(),
    ///    &a_value
    /// );
    ///
    /// assert_eq!(
    ///     a.eval(Some(&feed_dict)),
    ///     Array::from_vec(vec![0., 1., 2.], vec![1, 3])
    /// )
    /// ```
    pub fn new_placeholder(id: String, shape: Vec<usize>) -> Tensor<T> {
        Tensor {
            op: Rc::new(Placeholder::new(id, shape)),
            variable_data: None,
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
    /// let a = Tensor::new_variable(Array::from_vec(vec![0., 1., 2.], vec![1, 3]));
    ///
    /// assert_eq!(
    ///     a.eval(None),
    ///     Array::from_vec(vec![0., 1., 2.], vec![1, 3])
    /// )
    /// ```
    pub fn new_variable(init_value: Array<T>) -> Tensor<T> {
        let var_data = Rc::new(RefCell::new(init_value));
        Tensor {
            op: Rc::new(Variable::new(Rc::clone(&var_data))),
            variable_data: Some(var_data),
        }
    }

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
    /// let a = Tensor::new_variable(Array::from_vec(vec![0., 1., 2., 3.], vec![2, 2]));
    /// let b = Tensor::new_variable(Array::from_vec(vec![4., 5., 6., 7.], vec![2, 2]));
    /// let add = &a + &b; // `add` is a tensor!
    ///
    /// assert_eq!(
    ///     add.eval(None),
    ///     Array::from_vec(vec![4., 6., 8., 10.], vec![2, 2])
    /// )
    /// ```
    pub fn eval(&self, feed_dict: Option<&HashMap<String, &Array<T>>>) -> Array<T> {
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
    /// let a = Tensor::new_variable(Array::from_vec(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]));
    /// let b = Tensor::new_variable(Array::from_vec(vec![4., 5., 6.], vec![3, 1]));
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
        feed_dict: Option<&HashMap<String, &Array<T>>>,
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
    /// let a = Tensor::new_variable(Array::from_vec(vec![0., 1., 2., 3., 4., 5.], vec![2, 3]));
    /// let b = Tensor::new_variable(Array::from_vec(vec![4., 5., 6.], vec![3, 1]));
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
            variable_data: None,
        }
    }

    /// Updates stored variable's data by assigning a new data to it.
    ///
    /// Note that only tensors with `Variable` operator can be updated.
    ///
    /// * `new_value` - New value to be assigned.
    ///
    /// **Panics** if stored graph operator is not of `Variable` type.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let arr = Tensor::new_variable(Array::from_vec(vec![0., 1., 2., 3.], vec![2, 2]));
    ///
    /// arr.assign(&Array::new(1., vec![2, 2]));
    ///
    /// assert_eq!(
    ///     arr.eval(None),
    ///     Array::new(1., vec![2, 2])
    /// )
    /// ```
    pub fn assign(&self, new_value: &Array<T>) {
        *self
            .variable_data
            .as_ref()
            .expect("New data cannot be assigned to non-variable tensors.")
            .borrow_mut() = new_value.clone();
    }

    /// Updates stored variable's data by adding given data to it.
    ///
    /// Note that only tensors with `Variable` operator can be updated.
    ///
    /// * `new_value` - New value to be added to a current array.
    ///
    /// **Panics** if stored graph operator is not of `Variable` type.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    ///
    /// let arr = Tensor::new_variable(Array::from_vec(vec![0., 1., 2., 3.], vec![2, 2]));
    ///
    /// arr.assign_add(&Array::new(1., vec![2, 2]));
    ///
    /// assert_eq!(
    ///     arr.eval(None),
    ///     Array::from_vec(vec![1., 2., 3., 4.], vec![2, 2])
    /// )
    /// ```
    pub fn assign_add(&self, value: &Array<T>) {
        *self
            .variable_data
            .as_ref()
            .expect("New data cannot be added to non-variable tensors.")
            .borrow_mut() += value;
    }
}
