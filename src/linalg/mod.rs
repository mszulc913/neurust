mod array;
mod array_view;
mod matmul;
mod reduce;
mod utils;

use num::Float;
use std::fmt;

pub trait Numeric: Float + fmt::Display + Copy + fmt::Debug + 'static {}
impl<T> Numeric for T where T: Float + fmt::Display + Copy + fmt::Debug + 'static + PartialOrd {}

pub use array::*;
pub use array_view::ArrayView;
pub use reduce::{reduce, reduce_max, reduce_mean, reduce_min, reduce_prod, reduce_sum};
