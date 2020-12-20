use crate::graph::GraphOp;
use crate::linalg::Numeric;
use crate::Array;
use std::collections::HashMap;
use std::rc::Rc;

// Defines `GraphOp` for operators that applies some function to all
// elements of the input array.
macro_rules! impl_map_op {
    ($op_name:ident, $op_name_str:expr, $compute_fn:expr, $grad_fn:expr) => {
        pub(crate) struct $op_name<T: Numeric> {
            input: Rc<dyn GraphOp<T>>,
        }

        impl<T: Numeric> $op_name<T> {
            pub fn new(input: Rc<dyn GraphOp<T>>) -> $op_name<T> {
                $op_name { input }
            }
        }

        impl<'a, T: Numeric + 'static> GraphOp<T> for $op_name<T> {
            fn compute(
                &self,
                feed_dict: Option<&HashMap<String, &Array<T>>>,
                cache: &mut HashMap<usize, Array<T>>,
            ) -> Array<T> {
                let mut res = self.input.value(feed_dict, cache);
                res.map_assign($compute_fn);
                res
            }

            fn get_name(&self) -> &str {
                $op_name_str
            }

            fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
                Some(vec![Rc::clone(&self.input)])
            }

            fn compute_accumm_grad(
                &self,
                feed_dict: Option<&HashMap<String, &Array<T>>>,
                compute_cache: &mut HashMap<usize, Array<T>>,
                dependant_node: &dyn GraphOp<T>,
                grad: &Array<T>,
            ) -> Option<Array<T>> {
                if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
                    Some(grad * &self.input.value(feed_dict, compute_cache).map($grad_fn))
                } else {
                    None
                }
            }

            fn as_trait(&self) -> &dyn GraphOp<T> {
                self as &dyn GraphOp<T>
            }
        }
    };
}

fn sigmoid<T: Numeric>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}

fn sigmoid_derivative<T: Numeric>(x: T) -> T {
    let sigmoid_result = sigmoid(x);
    sigmoid_result * (T::one() - sigmoid_result)
}

impl_map_op!(SinOp, "SinOp", |x| x.sin(), |x| x.cos());
impl_map_op!(CosOp, "CosOp", |x| x.cos(), |x| -x.sin());
impl_map_op!(LnOp, "LnOp", |x| x.ln(), |x| T::one() / x);
impl_map_op!(SigmoidOp, "SigmoidOp", sigmoid, sigmoid_derivative);

// Defines `GraphOp` for operators that applies some parametrized function to all
// elements of the input array.
macro_rules! impl_map_op_with_parameter {
    ($op_name:ident, $op_name_str:expr, $compute_fn:expr, $grad_fn:expr) => {
        pub(crate) struct $op_name<T: Numeric> {
            input: Rc<dyn GraphOp<T>>,
            parameter: T,
        }

        impl<T: Numeric> $op_name<T> {
            pub fn new(input: Rc<dyn GraphOp<T>>, parameter: T) -> $op_name<T> {
                $op_name { input, parameter }
            }
        }

        impl<'a, T: Numeric + 'static> GraphOp<T> for $op_name<T> {
            fn compute(
                &self,
                feed_dict: Option<&HashMap<String, &Array<T>>>,
                cache: &mut HashMap<usize, Array<T>>,
            ) -> Array<T> {
                let mut res = self.input.value(feed_dict, cache);
                res.map_assign(|x| $compute_fn(x, self.parameter));
                res
            }

            fn get_name(&self) -> &str {
                $op_name_str
            }

            fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
                Some(vec![Rc::clone(&self.input)])
            }

            fn compute_accumm_grad(
                &self,
                feed_dict: Option<&HashMap<String, &Array<T>>>,
                compute_cache: &mut HashMap<usize, Array<T>>,
                dependant_node: &dyn GraphOp<T>,
                grad: &Array<T>,
            ) -> Option<Array<T>> {
                if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
                    Some(
                        grad * &self
                            .input
                            .value(feed_dict, compute_cache)
                            .map(|x| $grad_fn(x, self.parameter)),
                    )
                } else {
                    None
                }
            }

            fn as_trait(&self) -> &dyn GraphOp<T> {
                self as &dyn GraphOp<T>
            }
        }
    };
}

impl_map_op_with_parameter!(PowOp, "PowOp", |x: T, pow| x.powf(pow), |x: T, pow| pow
    * x.powf(pow - T::one()));
impl_map_op_with_parameter!(
    LogOp,
    "LogOp",
    |x: T, base: T| x.log(base),
    |x: T, base: T| T::one() / (x * base.ln())
);
