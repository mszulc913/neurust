mod array;
mod array_view;
mod matmul;
mod utils;

use num::Float;
use std::fmt;

pub trait Numeric: Float + fmt::Display + Copy + fmt::Debug + 'static {}
impl<T> Numeric for T where T: Float + fmt::Display + Copy + fmt::Debug + 'static {}

pub use array::*;
pub use array_view::ArrayView;
