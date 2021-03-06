use crate::graph::math::{CosOp, LnOp, LogOp, PowOp, ReLUOp, SigmoidOp, SinOp, TanhOp};
use crate::linalg::Numeric;
use crate::Tensor;
use std::rc::Rc;

pub fn sin<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(SinOp::new(Rc::clone(&tensor.op))))
}

pub fn cos<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(CosOp::new(Rc::clone(&tensor.op))))
}

pub fn ln<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(LnOp::new(Rc::clone(&tensor.op))))
}

pub fn sigmoid<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(SigmoidOp::new(Rc::clone(&tensor.op))))
}

pub fn pow<T: Numeric>(tensor: &Tensor<T>, pow: T) -> Tensor<T> {
    Tensor::new(Rc::new(PowOp::new(Rc::clone(&tensor.op), pow)))
}

pub fn log<T: Numeric>(tensor: &Tensor<T>, base: T) -> Tensor<T> {
    Tensor::new(Rc::new(LogOp::new(Rc::clone(&tensor.op), base)))
}

pub fn tanh<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(TanhOp::new(Rc::clone(&tensor.op))))
}

pub fn relu<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor::new(Rc::new(ReLUOp::new(Rc::clone(&tensor.op))))
}
