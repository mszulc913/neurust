use super::array::{Array, Slice};
use crate::linalg::Numeric;

/// Proxy structure for accessing `Array` data.
///
/// This structure is returned when slicing `Array`, i.e.
/// calling `.s()` method.
///
/// * `data` - Reference to `Array` data vector.
/// * `index` - Slice index vector.
/// * `shape` - Reference to `Array` shape vector.
#[derive(PartialEq, Debug)]
pub struct ArrayView<'a, T: Numeric> {
    data: &'a [T],
    index: Vec<Slice>,
    shape: &'a [usize],
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    /// Creates a new `ArrayView`.
    ///
    /// * `data` - Reference to `Array` data vector.
    /// * `index` - Slice index vector.
    /// * `shape` - Reference to `Array` shape vector.
    pub fn new(data: &'a [T], index: Vec<Slice>, shape: &'a [usize]) -> ArrayView<'a, T> {
        ArrayView { data, index, shape }
    }

    /// Converts `ArrayView` to `Array`.
    ///
    /// This copies values from the original object.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate neurust;
    /// use neurust::prelude::*;
    /// # fn main() {
    /// let arr = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 4]
    /// );
    /// // create array with first row only
    /// let arr_sliced = arr.s(s![0, ..]).to_array();
    /// # }
    /// ```
    pub fn to_array(&self) -> Array<T> {
        let mut data: Vec<T> = Vec::new();
        let mut curr_idx = vec![0; self.shape.len()];

        let shape_len = curr_idx.len();
        let mut new_shape = Vec::new();

        for i in 0..self.index.len() {
            match &self.index[i] {
                Slice::Index(_) => {}
                Slice::Range(range) => new_shape.push(range.end - range.start),
                Slice::RangeTo(range) => new_shape.push(range.end),
                Slice::RangeFrom(range) => new_shape.push(self.shape[i] - range.start),
                Slice::RangeFull(_) => new_shape.push(self.shape[i]),
            }
        }
        if new_shape.is_empty() {
            new_shape.push(1);
        }

        for val in self.data.iter() {
            // We iterate over all possible indices (curr_idx) and check if every element matches given index slice.
            let mut is_in_index = true;
            for (&curr_idx_elem, slice_idx_elem) in curr_idx.iter().zip(self.index.iter()) {
                match slice_idx_elem {
                    Slice::Range(range) => {
                        is_in_index &= range.start <= curr_idx_elem && curr_idx_elem < range.end
                    }
                    Slice::RangeTo(range) => is_in_index &= curr_idx_elem < range.end,
                    Slice::RangeFrom(range) => {
                        is_in_index &= range.start <= curr_idx_elem;
                    }
                    Slice::RangeFull(_) => {
                        is_in_index &= true;
                    }
                    Slice::Index(i) => {
                        is_in_index &= *i == curr_idx_elem;
                    }
                }
                if !is_in_index {
                    break;
                }
            }
            if is_in_index {
                data.push(*val);
            }
            curr_idx[shape_len - 1] += 1;
            for i in (0..shape_len).rev() {
                if curr_idx[i] == self.shape[i] && i != 0 {
                    curr_idx[i] = 0;
                    curr_idx[i - 1] += 1;
                }
            }
        }
        Array::<T>::from_vec(data, new_shape)
    }
}
