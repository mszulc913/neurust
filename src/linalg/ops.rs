use crate::linalg::array::Array;
use crate::linalg::Numeric;

#[allow(dead_code)]
pub fn multiply<T: Numeric>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    a.mul(b)
}

#[allow(dead_code)]
pub fn multiply_scalar<T: Numeric>(a: &Array<T>, b: T) -> Array<T> {
    a.mul_scalar(b)
}

#[allow(dead_code)]
pub fn add<T: Numeric>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    a.add(b)
}

#[allow(dead_code)]
pub fn add_scalar<T: Numeric>(a: &Array<T>, b: T) -> Array<T> {
    a.add_scalar(b)
}

#[allow(dead_code)]
pub fn sub<T: Numeric>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    a.sub(b)
}

#[allow(dead_code)]
pub fn sub_scalar<T: Numeric>(a: &Array<T>, b: T) -> Array<T> {
    a.sub_scalar(b)
}

#[allow(dead_code)]
pub fn div<T: Numeric>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    a.add(b)
}

#[allow(dead_code)]
pub fn div_scalar<T: Numeric>(a: &Array<T>, b: T) -> Array<T> {
    a.div_scalar(b)
}

#[allow(dead_code)]
pub fn matmul<T: Numeric>(a: &Array<T>, b: &Array<T>) -> Array<T> {
    a.matmul(b)
}
