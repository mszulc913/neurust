use crate::graph::math::{CosOp, LnOp, LogOp, PowOp, SigmoidOp, SinOp};
use crate::linalg::Numeric;
use crate::Tensor;
use std::rc::Rc;

pub fn sin<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor {
        op: Rc::new(SinOp::new(Rc::clone(&tensor.op))),
    }
}

pub fn cos<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor {
        op: Rc::new(CosOp::new(Rc::clone(&tensor.op))),
    }
}

pub fn ln<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor {
        op: Rc::new(LnOp::new(Rc::clone(&tensor.op))),
    }
}

pub fn sigmoid<T: Numeric>(tensor: &Tensor<T>) -> Tensor<T> {
    Tensor {
        op: Rc::new(SigmoidOp::new(Rc::clone(&tensor.op))),
    }
}

pub fn pow<T: Numeric>(tensor: &Tensor<T>, pow: T) -> Tensor<T> {
    Tensor {
        op: Rc::new(PowOp::new(Rc::clone(&tensor.op), pow)),
    }
}

pub fn log<T: Numeric>(tensor: &Tensor<T>, base: T) -> Tensor<T> {
    Tensor {
        op: Rc::new(LogOp::new(Rc::clone(&tensor.op), base)),
    }
}
