use crate::graph::GraphOp;
use crate::linalg::{reduce_mean, reduce_sum, Numeric};
use crate::Array;
use num::cast;
use std::collections::HashMap;
use std::rc::Rc;

// Implements `GraphOp` struct for reduction operations.
macro_rules! impl_struct_reduce_op {
    ($op_name:ident) => {
        pub struct $op_name<T: Numeric> {
            input: Rc<dyn GraphOp<T>>,
            axis: Option<usize>,
            keep_dims: bool,
        }
        impl<T: Numeric> $op_name<T> {
            pub fn new(
                input: Rc<dyn GraphOp<T>>,
                axis: Option<usize>,
                keep_dims: bool,
            ) -> $op_name<T> {
                $op_name {
                    input,
                    axis,
                    keep_dims,
                }
            }
        }
    };
}

// Implements `GraphOp` trait methods reduction operations except `compute`
// and `compute_acumm_gradient`.
macro_rules! impl_trait_reduce_op {
    ($op_name:ident, $op_name_str:expr) => {
        fn get_name(&self) -> &str {
            $op_name_str
        }

        fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
            Some(vec![Rc::clone(&self.input)])
        }

        fn as_trait(&self) -> &dyn GraphOp<T> {
            self as &dyn GraphOp<T>
        }
    };
}

impl_struct_reduce_op!(ReduceSumOp);
impl<'a, T: Numeric> GraphOp<T> for ReduceSumOp<T> {
    impl_trait_reduce_op!(ReduceSumOp, "ReduceSumOp");
    fn compute(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        reduce_sum(
            &self.input.value(feed_dict, cache),
            self.axis,
            self.keep_dims,
        )
    }

    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            Some(
                grad * &Array::new(
                    T::one(),
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_reduce_op!(ReduceMeanOp);
impl<'a, T: Numeric> GraphOp<T> for ReduceMeanOp<T> {
    impl_trait_reduce_op!(ReduceMeanOp, "ReduceMeanOp");
    fn compute(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        reduce_mean(
            &self.input.value(feed_dict, cache),
            self.axis,
            self.keep_dims,
        )
    }

    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            let computed_value = self.input.value(feed_dict, compute_cache);
            let dim_size = self.axis.unwrap_or(computed_value.data.len());
            Some(
                grad * &Array::new(
                    T::one() / cast::<_, T>(dim_size).unwrap(),
                    computed_value.get_shape(),
                ),
            )
        } else {
            None
        }
    }
}
