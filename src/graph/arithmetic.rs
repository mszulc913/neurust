use crate::graph::GraphOp;
use crate::linalg::{Array, Numeric};
use std::collections::HashMap;
use std::rc::Rc;

// Implements `GraphOp` struct for basic arithmetic operations with 2 node inputs.
macro_rules! impl_struct_op_2_inputs {
    ($op_name:ident) => {
        pub struct $op_name<T: Numeric> {
            input_1: Rc<dyn GraphOp<T>>,
            input_2: Rc<dyn GraphOp<T>>,
        }
        impl<T: Numeric> $op_name<T> {
            pub fn new(input_1: Rc<dyn GraphOp<T>>, input_2: Rc<dyn GraphOp<T>>) -> $op_name<T> {
                $op_name { input_1, input_2 }
            }
        }
    };
}

// Implements `GraphOp` struct for basic arithmetic operations with 1 node input and a scalar.
macro_rules! impl_struct_op_1_input_scalar {
    ($op_name:ident) => {
        pub struct $op_name<T: Numeric> {
            input: Rc<dyn GraphOp<T>>,
            scalar: T,
        }
        impl<T: Numeric> $op_name<T> {
            pub fn new(input: Rc<dyn GraphOp<T>>, scalar: T) -> $op_name<T> {
                $op_name { input, scalar }
            }
        }
    };
}

// Implements `GraphOp` trait methods for basic arithmetic operations with 2 node inputs.
macro_rules! impl_trait_op_2_inputs {
    ($op_name:ident, $op_name_str:expr, $op_token:tt) => {
        fn compute(
            &self,
            feed_dict: Option<&HashMap<String, &Array<T>>>,
            cache: &mut HashMap<usize, Array<T>>
        ) -> Array<T> {
            &self.input_1.value(feed_dict, cache) $op_token &self.input_2.value(feed_dict, cache)
        }

        fn get_name(&self) -> &str {
            $op_name_str
        }

        fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>>{
            Some(vec![Rc::clone(&self.input_1), Rc::clone(&self.input_2)])
        }

        fn as_trait(&self) -> &dyn GraphOp<T> {
            self as &dyn GraphOp<T>
        }
    };
}

// Implements `GraphOp` trait methods for basic arithmetic operations with 1 node input and scalar.
macro_rules! impl_trait_op_1_input_scalar {
    ($op_name:ident, $op_name_str:expr, $op_token:tt) => {
        fn compute(
            &self,
            feed_dict: Option<&HashMap<String, &Array<T>>>,
            cache: &mut HashMap<usize, Array<T>>
        ) -> Array<T> {
            &self.input.value(feed_dict, cache) $op_token self.scalar
        }

        fn get_name(&self) -> &str {
            $op_name_str
        }

        fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>>{
            Some(vec![Rc::clone(&self.input)])
        }

        fn as_trait(&self) -> &dyn GraphOp<T> {
            self as &dyn GraphOp<T>
        }
    };
}

impl_struct_op_2_inputs!(AddOp);
impl<'a, T: Numeric> GraphOp<T> for AddOp<T> {
    impl_trait_op_2_inputs!(AddOp, "AddOp", +);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input_1.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    T::one(),
                    self.input_1.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else if dependant_node.ref_as_usize() == self.input_2.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    T::one(),
                    self.input_2.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_2_inputs!(MulOp);
impl<'a, T: Numeric> GraphOp<T> for MulOp<T> {
    impl_trait_op_2_inputs!(MulOp, "MulOp", *);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input_1.ref_as_usize() {
            Some(grad * &self.input_2.value(feed_dict, compute_cache))
        } else if dependant_node.ref_as_usize() == self.input_2.ref_as_usize() {
            Some(grad * &self.input_1.value(feed_dict, compute_cache))
        } else {
            None
        }
    }
}

impl_struct_op_2_inputs!(SubOp);
impl<'a, T: Numeric> GraphOp<T> for SubOp<T> {
    impl_trait_op_2_inputs!(SubOp, "SubOp", -);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input_1.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    -T::one(),
                    self.input_2.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else if dependant_node.ref_as_usize() == self.input_2.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    -T::one(),
                    self.input_1.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_2_inputs!(DivOp);
impl<'a, T: Numeric> GraphOp<T> for DivOp<T> {
    impl_trait_op_2_inputs!(DivOp, "DivOp", /);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input_1.ref_as_usize() {
            let value1 = self.input_1.value(feed_dict, compute_cache);
            let value2 = self.input_2.value(feed_dict, compute_cache);
            let ones = Array::<T>::new(T::one(), value1.get_shape());
            Some(grad * &(&ones / &value2))
        } else if dependant_node.ref_as_usize() == self.input_2.ref_as_usize() {
            let value1 = self.input_1.value(feed_dict, compute_cache);
            let value2 = self.input_2.value(feed_dict, compute_cache);
            let minus_ones = Array::<T>::new(-T::one(), value1.get_shape());
            Some(grad * &(&minus_ones / &(&value2 * &value2)))
        } else {
            None
        }
    }
}

impl_struct_op_1_input_scalar!(AddScalarOp);
impl<'a, T: Numeric> GraphOp<T> for AddScalarOp<T> {
    impl_trait_op_1_input_scalar!(AddScalarOp, "AddScalarOp", +);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    T::one(),
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_1_input_scalar!(SubScalarOp);
impl<'a, T: Numeric> GraphOp<T> for SubScalarOp<T> {
    impl_trait_op_1_input_scalar!(SubScalarOp, "SubScalarOp", -);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    -T::one(),
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_1_input_scalar!(MulScalarOp);
impl<'a, T: Numeric> GraphOp<T> for MulScalarOp<T> {
    impl_trait_op_1_input_scalar!(MulScalarOp, "MulScalarOp", *);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    self.scalar,
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_1_input_scalar!(DivScalarOp);
impl<'a, T: Numeric> GraphOp<T> for DivScalarOp<T> {
    impl_trait_op_1_input_scalar!(DivScalarOp, "DivScalarOp", /);
    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input.ref_as_usize() {
            Some(
                grad * &Array::<T>::new(
                    T::one() / self.scalar,
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }
}

impl_struct_op_2_inputs!(MatMulOp);
impl<'a, T: Numeric> GraphOp<T> for MatMulOp<T> {
    fn compute(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        (&self.input_1.value(feed_dict, cache)).matmul(&self.input_2.value(feed_dict, cache))
    }

    fn compute_accum_grad(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        if dependant_node.ref_as_usize() == self.input_1.ref_as_usize() {
            Some(grad.matmul(&self.input_2.value(feed_dict, compute_cache).transpose()))
        } else if dependant_node.ref_as_usize() == self.input_2.ref_as_usize() {
            Some(
                self.input_1
                    .value(feed_dict, compute_cache)
                    .transpose()
                    .matmul(&grad),
            )
        } else {
            None
        }
    }

    fn get_name(&self) -> &str {
        "MatMulOp"
    }

    fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
        Some(vec![Rc::clone(&self.input_1), Rc::clone(&self.input_2)])
    }

    fn as_trait(&self) -> &dyn GraphOp<T> {
        self as &dyn GraphOp<T>
    }
}

pub struct NegOp<T: Numeric> {
    input: Rc<dyn GraphOp<T>>,
}
impl<T: Numeric> NegOp<T> {
    pub fn new(input: Rc<dyn GraphOp<T>>) -> NegOp<T> {
        NegOp { input }
    }
}

impl<'a, T: Numeric> GraphOp<T> for NegOp<T> {
    fn compute(
        &self,
        feed_dict: Option<&HashMap<String, &Array<T>>>,
        cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        self.input.value(feed_dict, cache).neg()
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
                    -T::one(),
                    self.input.value(feed_dict, compute_cache).get_shape(),
                ),
            )
        } else {
            None
        }
    }

    fn get_name(&self) -> &str {
        "NegOp"
    }

    fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
        Some(vec![Rc::clone(&self.input)])
    }

    fn as_trait(&self) -> &dyn GraphOp<T> {
        self as &dyn GraphOp<T>
    }
}
