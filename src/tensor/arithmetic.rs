use crate::graph::arithmetic::{
    AddOp, AddScalarOp, DivOp, DivScalarOp, MulOp, MulScalarOp, NegOp, SubOp, SubScalarOp,
};
use crate::linalg::Numeric;
use crate::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

macro_rules! impl_tensor_operators_overload_2_inputs {
    ($op_name:ident, $op_method_name:ident, $graph_op_name:ident) => {
        impl<T: Numeric> $op_name<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: &Tensor<T>) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(
                        Rc::clone(&self.op),
                        Rc::clone(&other.op),
                    )),
                }
            }
        }

        impl<T: Numeric> $op_name<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: Tensor<T>) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(
                        Rc::clone(&self.op),
                        Rc::clone(&other.op),
                    )),
                }
            }
        }

        impl<T: Numeric> $op_name<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: Tensor<T>) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(
                        Rc::clone(&self.op),
                        Rc::clone(&other.op),
                    )),
                }
            }
        }

        impl<T: Numeric> $op_name<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: &Tensor<T>) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(
                        Rc::clone(&self.op),
                        Rc::clone(&other.op),
                    )),
                }
            }
        }
    };
}

impl_tensor_operators_overload_2_inputs!(Add, add, AddOp);
impl_tensor_operators_overload_2_inputs!(Sub, sub, SubOp);
impl_tensor_operators_overload_2_inputs!(Div, div, DivOp);
impl_tensor_operators_overload_2_inputs!(Mul, mul, MulOp);

macro_rules! impl_tensor_operators_overload_with_scalar {
    ($op_name:ident, $op_method_name:ident, $graph_op_name:ident) => {
        impl<T: Numeric> $op_name<T> for &Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: T) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&self.op), other)),
                }
            }
        }

        impl $op_name<Tensor<f32>> for f32 {
            type Output = Tensor<f32>;
            fn $op_method_name(self, other: Tensor<f32>) -> Tensor<f32> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&other.op), self)),
                }
            }
        }

        impl $op_name<&Tensor<f32>> for f32 {
            type Output = Tensor<f32>;
            fn $op_method_name(self, other: &Tensor<f32>) -> Tensor<f32> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&other.op), self)),
                }
            }
        }

        impl $op_name<Tensor<f64>> for f64 {
            type Output = Tensor<f64>;
            fn $op_method_name(self, other: Tensor<f64>) -> Tensor<f64> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&other.op), self)),
                }
            }
        }

        impl $op_name<&Tensor<f64>> for f64 {
            type Output = Tensor<f64>;
            fn $op_method_name(self, other: &Tensor<f64>) -> Tensor<f64> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&other.op), self)),
                }
            }
        }

        impl<T: Numeric> $op_name<T> for Tensor<T> {
            type Output = Tensor<T>;
            fn $op_method_name(self, other: T) -> Tensor<T> {
                Tensor {
                    op: Rc::new($graph_op_name::new(Rc::clone(&self.op), other)),
                }
            }
        }
    };
}

impl_tensor_operators_overload_with_scalar!(Add, add, AddScalarOp);
impl_tensor_operators_overload_with_scalar!(Sub, sub, SubScalarOp);
impl_tensor_operators_overload_with_scalar!(Mul, mul, MulScalarOp);
impl_tensor_operators_overload_with_scalar!(Div, div, DivScalarOp);

impl<T: Numeric> Neg for &Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        Tensor {
            op: Rc::new(NegOp::new(Rc::clone(&self.op))),
        }
    }
}

impl<T: Numeric> Neg for Tensor<T> {
    type Output = Tensor<T>;
    fn neg(self) -> Tensor<T> {
        Tensor {
            op: Rc::new(NegOp::new(Rc::clone(&self.op))),
        }
    }
}
