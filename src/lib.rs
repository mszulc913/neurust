pub mod graph;
pub mod linalg;
pub mod prelude;
pub mod tensor;

pub use linalg::{Array, Slice};
pub use tensor::{reduce_mean, reduce_sum, Tensor};
