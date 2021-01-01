use super::array_view::ArrayView;
use super::utils::{check_shape_positive, transpose_2d_matrix_slices};
use crate::linalg::broadcast::BroadcastIterator;
use crate::linalg::matmul::matmul_2d_matrix_slices;
use crate::linalg::utils::{get_shape_after_broadcast, get_shape_after_broadcast_matmul};
use crate::linalg::Numeric;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Range, RangeFrom,
    RangeFull, RangeTo, Sub, SubAssign,
};

/// N-dimensional array.
///
/// Supports overloaded arithmetic operators and broadcasting operands.
///
/// * `shape` - `Vec<usize>` with matrix' shape. For example 2D matrix has a shape of [x, y].
/// * `data` - `Vec<T>` with matrix' data.
#[derive(PartialEq)]
pub struct Array<T: Numeric> {
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<T>,
}

impl<T: Numeric> Array<T> {
    /// Creates a new `Array`.
    ///
    /// Created array has shape `shape` and is initialized with `init_value`.
    /// `init_value` also indicates what data type `T` is stored inside the array.
    /// `T` should be of floating point type.
    ///
    /// * `shape` - Non-zero `Shape` of an array.
    /// * `init_value` - Initial value of type `T` array will be populated with.
    ///
    /// **Panics** if `shape` contains zero.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let arr = Array::new(4., vec![3, 2, 2]);
    /// ```
    pub fn new(init_value: T, shape: Vec<usize>) -> Self {
        check_shape_positive(&shape);
        let size = shape.iter().product();

        Self {
            shape,
            data: vec![init_value; size],
        }
    }

    /// Creates a new `Array` from one-dimensional vector.
    ///
    /// Shape of the array must match the length of `data`, meaning
    /// the length must be equal to product of all dimensions.
    ///
    /// * `data` - Vector of type `T` with data to be used.
    /// * `shape`- Non-zero `Shape` of the array.
    ///
    /// **Panics** if `shape` contains zero or `data` and `shape` are incompatible.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let arr = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 4]
    /// );
    /// ```
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Array<T> {
        check_shape_positive(&shape);
        if data.len() != shape.iter().product() {
            panic!(
                "Incompatible shapes! Data has length of {} and given shape is: {:?}",
                data.len(),
                shape
            )
        }
        Array { data, shape }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// *Slices* an array.
    ///
    /// This allows to access specific array region and to extract sub-arrays.
    /// It is basically more general indexing operator. It works similar to `Numpy` slices with
    /// the difference that result is immutable and negative indices aren't supported yet.
    ///
    /// Each `index` element corresponds to a single dimension from `self.shape` vector.
    ///
    /// There is more convienient way for specyfing `index` vector:
    /// `s!` macro. It supports `usize` values and supports the following
    /// formats of range specification: `x..y`, `..`, `x..`, `..x`.
    ///
    /// * `index` - Slice index as vector of `Slice` enums. Length of this
    /// vector must be the same as length of the `self.shape` vector.
    ///
    /// **Panics** if slice vector has wrong length or index values are out of bounds.
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
    /// // fetch first row and second and third columns
    /// let slice = vec![Slice::Index(0), Slice::Range(1..3)];
    /// let sliced_arr = arr.s(slice);
    ///
    /// // the same using s! macro
    /// let sliced_arr_macro = arr.s(s![0, 1..3]);
    ///
    /// assert_eq!(sliced_arr, sliced_arr_macro);
    /// # }
    /// ```
    pub fn s(&self, index: Vec<Slice>) -> ArrayView<T> {
        if index.len() != self.shape.len() {
            panic!(
                "
                Given index has invalid length. Expected: {}, actual: {}",
                self.shape.len(),
                index.len()
            );
        }
        for i in 0..self.shape.len() {
            if let Slice::Index(idx) = index[i] {
                if idx >= self.shape[i] {
                    panic!(
                        "Index out of bounds. Got index {:?} for shape {:?}",
                        self.shape, index
                    );
                }
            }
        }
        ArrayView::<T>::new(&self.data, index, &self.shape)
    }

    /// Returns value at a given position.
    ///
    /// * `index` - Index of the same length as `self.shape` vector.
    ///
    /// **Panics** if `index` has wrong length or values of `index` are
    /// out of bounds.
    ///
    /// # Examples
    /// ```
    /// use neurust::prelude::*;
    /// let arr = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// assert_eq!(arr.i(vec![1, 0, 2]), 7.);
    /// ```
    pub fn i(&self, index: Vec<usize>) -> T {
        self[index]
    }

    // Checks if index has proper length and values. Panics if not.
    fn check_index(&self, index: &[usize]) {
        if index.len() != self.shape.len() {
            panic!(
                "
                Given index has invalid length. Expected: {}, actual: {}",
                self.shape.len(),
                index.len()
            );
        }
        for i in 0..self.shape.len() {
            if index[i] >= self.shape[i] {
                panic!(
                    "Index out of bounds. Got index {:?} for shape {:?}",
                    self.shape, index
                )
            }
        }
    }

    // Translates vectorized index into single scalar index.
    fn compute_data_index(&self, index: &[usize]) -> usize {
        let mut idx = 0;
        let mut prod = 1;
        for i in (0..self.shape.len()).rev() {
            idx += index[i] * prod;
            prod *= self.shape[i];
        }
        idx
    }

    // Creates a new array with elements being a function of paired elements
    // from current array and from other array. Operation can be broadcasted.
    fn compute_elementwise_with_other_array(&self, other: &Array<T>, f: fn(T, T) -> T) -> Array<T> {
        let shape = get_shape_after_broadcast(&self.shape, &other.shape);
        let mut data = vec![T::zero(); shape.iter().product()];

        self.compute_elementwise_with_other_array_on_mem_buffer(other, f, &shape, &mut data);

        Array { shape, data }
    }

    // Computes result of applying some function in a broadcasted way to given memory buffer.
    fn compute_elementwise_with_other_array_on_mem_buffer(
        &self,
        other: &Array<T>,
        f: fn(T, T) -> T,
        shape: &[usize],
        buff: &mut Vec<T>,
    ) {
        let mut trailing_dims = 0;
        for (&x, &y) in self.shape.iter().rev().zip(other.shape.iter().rev()) {
            if x == y {
                trailing_dims += 1
            } else {
                break;
            }
        }

        let slice_len: usize = shape[(shape.len() - trailing_dims)..].iter().product();
        for (i, (slice1, slice2)) in BroadcastIterator::new(self, other, trailing_dims).enumerate()
        {
            let output_slice = buff[(slice_len * i)..(slice_len * (i + 1))].as_mut();
            for (j, (elem1, elem2)) in slice1.iter().zip(slice2.iter()).enumerate() {
                *output_slice.get_mut(j).unwrap() = f(*elem1, *elem2);
            }
        }
    }

    // Creates a new array with elements being a function of elements
    // from current array and a scalar value. Operation can be broadcasted.
    fn compute_elementwise_with_scalar(&self, other: T, f: fn(T, T) -> T) -> Array<T> {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        for elem in &mut data {
            *elem = f(*elem, other);
        }
        Array { shape, data }
    }

    /// Creates a new array by applying given function to all
    /// stored elements.
    ///
    /// * `f` - Function to be applied.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.map(|x| 2. * x);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn map(&self, f: impl Fn(T) -> T) -> Array<T> {
        let mut data = self.data.clone();
        let shape = self.shape.clone();
        for elem in &mut data {
            *elem = f(*elem);
        }
        Array { shape, data }
    }

    /// Modifies an array by applying given function to all stored elements.
    ///
    /// * `f` - Function to be applied.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.map_assign(|x| 2. * x);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn map_assign(&mut self, f: impl Fn(T) -> T) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i]);
        }
    }

    // Updates array's elements to be a function of paired elements
    // from the array and from some other Array.
    fn assign_compute_elementwise_with_other_array(&mut self, other: &Array<T>, f: fn(T, T) -> T) {
        let shape = get_shape_after_broadcast(&self.shape, &other.shape);
        let mut data = vec![T::zero(); shape.iter().product()];

        self.compute_elementwise_with_other_array_on_mem_buffer(other, f, &shape, &mut data);

        self.data = data;
        self.shape = shape;
    }

    // Updates array's elements to be a function of elements from an array and a scalar value.
    fn assign_compute_elementwise_with_scalar(&mut self, other: T, f: fn(T, T) -> T) {
        for i in 0..self.data.len() {
            self.data[i] = f(self.data[i], other);
        }
    }

    /// Creates new `Array` with elements being a negation of the elements from
    /// the original array.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.neg();
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![-1., -2., -3., -4., -5., -6., -7., -8.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn neg(&self) -> Array<T> {
        self.map(|x| -x)
    }

    /// Modifies an array's elements by applying negation operator to them.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.neg_assign();
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![-1., -2., -3., -4., -5., -6., -7., -8.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn neg_assign(&mut self) {
        self.map_assign(|x| -x)
    }

    /// Computes addition of an array and some other array.
    ///
    /// Returns a new array.
    ///
    /// * `other` - Other array to be added.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.add(&b);
    /// result == Array::from_vec(
    ///         vec![3., 5., 7., 9., 11., 13., 15., 17.],
    ///         vec![2, 1, 4]
    ///     );
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![3., 5., 7., 9., 11., 13., 15., 17.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn add(&self, other: &Array<T>) -> Array<T> {
        self.compute_elementwise_with_other_array(other, |x, y| x + y)
    }

    /// Computes addition of an array and a scalar value.
    ///
    /// * `other` - Scalar value to be added.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.add_scalar(4.);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![5., 6., 7., 8., 9., 10., 11., 12.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn add_scalar(&self, other: T) -> Array<T> {
        self.compute_elementwise_with_scalar(other, |x, y| x + y)
    }

    /// Computes subtraction of some array from a current array.
    ///
    /// Returns a new array.
    ///
    /// * `other` - Other `Array` to be subtracted.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.sub(&b);
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![-1., -1., -1., -1., -1., -1., -1., -1.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn sub(&self, other: &Array<T>) -> Array<T> {
        self.compute_elementwise_with_other_array(other, |x, y| x - y)
    }

    /// Computes subtraction of a scalar value from a current array.
    ///
    /// * `other` - Scalar value to be subtracted.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.sub_scalar(1.);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![0., 1., 2., 3., 4., 5., 6., 7.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn sub_scalar(&self, other: T) -> Array<T> {
        self.compute_elementwise_with_scalar(other, |x, y| x - y)
    }

    /// Returns a new array being a product of element-wise multiplication of
    /// a current array and some other array.
    ///
    /// * `other` - Second array.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.mul(&b);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![2., 6., 12., 20., 30., 42., 56., 72.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn mul(&self, other: &Array<T>) -> Array<T> {
        self.compute_elementwise_with_other_array(other, |x, y| x * y)
    }

    /// Returns a new as a product of multiplication of a current array and a scalar value.
    ///
    /// * `other` - Scalar value of type `T`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.mul_scalar(2.);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn mul_scalar(&self, other: T) -> Array<T> {
        self.compute_elementwise_with_scalar(other, |x, y| x * y)
    }

    /// Element-wise division of two arrays.
    ///
    /// * `other` - Second array.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.div(&b);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![2., 2., 2., 2., 2., 2., 2., 2.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn div(&self, other: &Array<T>) -> Array<T> {
        self.compute_elementwise_with_other_array(other, |x, y| x / y)
    }

    /// Divides the array by a scalar value.
    ///
    /// * `other` - Scalar value of type `T`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// let result = a.div_scalar(2.);
    ///
    /// assert_eq!(
    ///     result,
    ///     Array::from_vec(
    ///         vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn div_scalar(&self, other: T) -> Array<T> {
        self.compute_elementwise_with_scalar(other, |x, y| x / y)
    }

    /// Adds elements from some other array to a current array.
    ///
    /// * `other` - Other array to be added.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.add_assign(&b);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![3., 5., 7., 9., 11., 13., 15., 17.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn add_assign(&mut self, other: &Array<T>) {
        self.assign_compute_elementwise_with_other_array(other, |x, y| x + y)
    }

    /// Adds a scalar value to a current array.
    ///
    /// * `other` - Scalar value to be added.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.add_assign_scalar(4.);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![5., 6., 7., 8., 9., 10., 11., 12.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn add_assign_scalar(&mut self, other: T) {
        self.assign_compute_elementwise_with_scalar(other, |x, y| x + y)
    }

    /// Subtracts elements of some other array from current array.
    ///
    /// * `other` - Other array to be subtracted.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.sub_assign(&b);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![-1., -1., -1., -1., -1., -1., -1., -1.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn sub_assign(&mut self, other: &Array<T>) {
        self.assign_compute_elementwise_with_other_array(other, |x, y| x - y)
    }

    /// Subtracts a scalar value from a current array.
    ///
    /// * `other` - Scalar value to be subtracted.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.sub_assign_scalar(1.);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![0., 1., 2., 3., 4., 5., 6., 7.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn sub_assign_scalar(&mut self, other: T) {
        self.assign_compute_elementwise_with_scalar(other, |x, y| x - y)
    }

    /// Performs in-place element-wise multiplication of a current array and some
    /// other array.
    ///
    /// * `other` - Second array.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![2., 3., 4., 5., 6., 7., 8., 9.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.mul_assign(&b);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![2., 6., 12., 20., 30., 42., 56., 72.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn mul_assign(&mut self, other: &Array<T>) {
        self.assign_compute_elementwise_with_other_array(other, |x, y| x * y)
    }

    /// Multiplies current array with a scalar.
    ///
    /// * `other` - Scalar value of type `T`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.mul_assign_scalar(2.);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn mul_assign_scalar(&mut self, other: T) {
        self.assign_compute_elementwise_with_scalar(other, |x, y| x * y)
    }

    /// Performs in-place element-wise division by some other array.
    ///
    /// * `other` - Second array.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///     vec![2, 1, 4]
    /// );
    /// let b = Array::from_vec(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.div_assign(&b);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![2., 2., 2., 2., 2., 2., 2., 2.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn div_assign(&mut self, other: &Array<T>) {
        self.assign_compute_elementwise_with_other_array(other, |x, y| x / y)
    }

    /// Divides the array by a scalar.
    ///
    /// * `other` - Scalar value of type `T`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![2., 4., 6., 8., 10., 12., 14., 16.],
    ///     vec![2, 1, 4]
    /// );
    ///
    /// a.div_assign_scalar(2.);
    ///
    /// assert_eq!(
    ///     a,
    ///     Array::from_vec(
    ///         vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///         vec![2, 1, 4]
    ///     )
    /// );
    /// ```
    pub fn div_assign_scalar(&mut self, other: T) {
        self.assign_compute_elementwise_with_scalar(other, |x, y| x / y)
    }

    /// Computes matrix product of two mutlidimensional arrays.
    ///
    /// Arrays can be multiplied only if:
    /// * their shapes (except last 2 dimensions) are valid in terms of array broadcasting,
    /// * they are at least 2 dimensional,
    /// * last two dimensions of both matrices are valid for matrix multiplication.
    ///
    /// Arrays are multiplied in a such way that matrix multiplication operator
    /// is applied to last 2 dimensions of the arrays.
    ///
    /// * `other` - Second array.
    ///
    /// **Panics** if both arrays don't have valid shapes in terms of array broadcasting
    /// and matrix product.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![
    ///         1., 2., 3.,
    ///         4., 5., 6.,
    ///
    ///         1., 2., 3.,
    ///         3., 2., 1.,
    ///     ],
    ///     vec![2, 2, 3]
    /// );
    /// let b = Array::from_vec(
    ///     vec![
    ///         1., 2.,
    ///         3., 4.,
    ///         5., 6.,
    ///
    ///         1., 2.,
    ///         3., 4.,
    ///         5., 6.
    ///     ],
    ///     vec![2, 3, 2]
    /// );
    ///
    /// let result = a.matmul(&b);
    ///
    /// assert_eq![
    ///     result,
    ///     Array::from_vec(
    ///         vec![
    ///             22., 28.,
    ///             49., 64.,
    ///
    ///             22., 28.,
    ///             14., 20.
    ///         ],
    ///         vec![2, 2, 2]
    ///     )
    /// ];
    /// ```
    pub fn matmul(&self, other: &Array<T>) -> Array<T> {
        let new_shape = get_shape_after_broadcast_matmul(&self.shape, &other.shape);

        let matrix1_shape = (
            self.shape[self.shape.len() - 2],
            self.shape[self.shape.len() - 1],
        );
        let matrix2_shape = (
            other.shape[other.shape.len() - 2],
            other.shape[other.shape.len() - 1],
        );

        let slice_len_output = matrix1_shape.0 * matrix2_shape.1;

        let data_len = new_shape.iter().product();
        let mut data = vec![T::zero(); data_len];

        for (i, (slice1, slice2)) in BroadcastIterator::new(self, other, 2).enumerate() {
            matmul_2d_matrix_slices(
                slice1,
                matrix1_shape.0,
                matrix1_shape.1,
                slice2,
                matrix2_shape.0,
                matrix2_shape.1,
                &mut data[(i * slice_len_output)..((i + 1) * slice_len_output)],
            )
        }
        Array {
            data,
            shape: new_shape,
        }
    }

    /// Transposes an array.
    ///
    /// Arrays can be transposed only if they are at least 2 dimensional.
    /// Only the last two dimensions are being transposed, i.e
    /// given an array of shape `[a, b, ..., d, e, g]`
    /// the resulting array will have shape `[a, b, ..., d, g, e]`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let a = Array::from_vec(
    ///     vec![
    ///         1., 2., 3.,
    ///         4., 5., 6.,
    ///
    ///         1., 2., 3.,
    ///         3., 2., 1.,
    ///     ],
    ///     vec![2, 2, 3]
    /// );
    ///
    /// let result = a.transpose();
    ///
    /// assert_eq![
    ///     result,
    ///     Array::from_vec(
    ///         vec![
    ///             1., 4.,
    ///             2., 5.,
    ///             3., 6.,
    ///
    ///             1., 3.,
    ///             2., 2.,
    ///             3., 1.,
    ///         ],
    ///         vec![2, 3, 2]
    ///     )
    /// ];
    /// ```
    pub fn transpose(&self) -> Array<T> {
        let new_shape = self.get_transposed_shape();
        let mut data = vec![T::zero(); self.shape.iter().product()];
        self.transpose_on_mem_buffer(&mut data);
        Array {
            data,
            shape: new_shape,
        }
    }

    // Computes a shape of a transposed array.
    fn get_transposed_shape(&self) -> Vec<usize> {
        if self.shape.len() < 2 {
            panic!(
                "Array with less than 2 dimensions cannot be transposed. Got shape: {:?}.",
                self.shape
            )
        }

        let mut new_shape = self.shape.clone();
        new_shape[self.shape.len() - 2] = self.shape[self.shape.len() - 1];
        new_shape[self.shape.len() - 1] = self.shape[self.shape.len() - 2];

        new_shape
    }

    // Transposes the array in a given memory.
    fn transpose_on_mem_buffer(&self, output_buffer: &mut Vec<T>) {
        let matrix_shape = (
            self.shape[self.shape.len() - 2],
            self.shape[self.shape.len() - 1],
        );

        let slice_len = matrix_shape.0 * matrix_shape.1;

        let num_slices = output_buffer.len() / slice_len;
        for i in 0..num_slices {
            transpose_2d_matrix_slices(
                &self.data[(i * slice_len)..((i + 1) * slice_len)],
                matrix_shape.0,
                matrix_shape.1,
                &mut output_buffer[(i * slice_len)..((i + 1) * slice_len)],
            )
        }
    }

    /// Transposes the array in place.
    ///
    /// Arrays can be transposed only if they are at least 2 dimensional.
    /// Only the last two dimensions are being transposed, i.e
    /// given an array of shape `[a, b, ..., d, e, g]`
    /// the resulting array will have shape `[a, b, ..., d, g, e]`.
    ///
    /// # Examples
    /// ```
    /// use neurust::linalg::Array;
    ///
    /// let mut a = Array::from_vec(
    ///     vec![
    ///         1., 2., 3.,
    ///         4., 5., 6.,
    ///
    ///         1., 2., 3.,
    ///         3., 2., 1.,
    ///     ],
    ///     vec![2, 2, 3]
    /// );
    ///
    /// a.transpose_assign();
    ///
    /// assert_eq![
    ///     a,
    ///     Array::from_vec(
    ///         vec![
    ///             1., 4.,
    ///             2., 5.,
    ///             3., 6.,
    ///
    ///             1., 3.,
    ///             2., 2.,
    ///             3., 1.,
    ///         ],
    ///         vec![2, 3, 2]
    ///     )
    /// ];
    /// ```
    pub fn transpose_assign(&mut self) {
        let mut data = vec![T::zero(); self.shape.iter().product()];
        self.transpose_on_mem_buffer(&mut data);
        self.data = data;
        self.shape = self.get_transposed_shape();
    }
}

impl<T: Numeric> fmt::Display for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, x) in self.data.iter().enumerate() {
            let mut prod = 1;
            let mut opened_brackets = 0;
            let mut closed_brackets = 0;
            for shape in self.shape.iter().rev() {
                prod *= shape;
                if i % prod == 0 {
                    opened_brackets += 1;
                }
            }
            if opened_brackets > 0 {
                for _i in 0..self.shape.len() - opened_brackets {
                    write!(f, " ")?;
                }
                for _i in 0..opened_brackets {
                    write!(f, "[")?;
                }
            }
            if (i + 1) % self.shape[self.shape.len() - 1] != 0 {
                write!(f, "{}, ", x)?;
            } else {
                write!(f, "{}", x)?;
            }

            prod = 1;
            for shape in self.shape.iter().rev() {
                prod *= shape;
                if (i + 1) % prod == 0 {
                    closed_brackets += 1;
                    write!(f, "]")?;
                }
            }
            if closed_brackets != self.shape.len() {
                for _i in 0..closed_brackets {
                    writeln!(f)?;
                }
            }
        }
        writeln!(f, " shape={:?}", self.shape)
    }
}

impl<T: Numeric> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<T: Numeric> Index<Vec<usize>> for Array<T> {
    type Output = T;
    fn index(&self, index: Vec<usize>) -> &Self::Output {
        self.check_index(&index);
        let idx = self.compute_data_index(&index);
        &self.data[idx]
    }
}

impl<T: Numeric> IndexMut<Vec<usize>> for Array<T> {
    fn index_mut(&mut self, index: Vec<usize>) -> &mut Self::Output {
        self.check_index(&index);
        let idx = self.compute_data_index(&index);
        &mut self.data[idx]
    }
}

impl<T: Numeric> Add<&Array<T>> for &Array<T> {
    type Output = Array<T>;
    fn add(self, other: &Array<T>) -> Array<T> {
        Array::add(self, other)
    }
}

impl<T: Numeric> Neg for &Array<T> {
    type Output = Array<T>;
    fn neg(self) -> Array<T> {
        Array::neg(self)
    }
}

impl<T: Numeric> AddAssign<&Array<T>> for Array<T> {
    fn add_assign(&mut self, other: &Array<T>) {
        self.add_assign(other);
    }
}

impl<T: Numeric> Sub<&Array<T>> for &Array<T> {
    type Output = Array<T>;
    fn sub(self, other: &Array<T>) -> Array<T> {
        Array::sub(self, other)
    }
}

impl<T: Numeric> SubAssign<&Array<T>> for Array<T> {
    fn sub_assign(&mut self, other: &Array<T>) {
        self.sub_assign(other);
    }
}

impl<T: Numeric> Mul<&Array<T>> for &Array<T> {
    type Output = Array<T>;
    fn mul(self, other: &Array<T>) -> Array<T> {
        Array::mul(self, other)
    }
}

impl<T: Numeric> MulAssign<&Array<T>> for Array<T> {
    fn mul_assign(&mut self, other: &Array<T>) {
        self.mul_assign(other);
    }
}

impl<T: Numeric> Div<&Array<T>> for &Array<T> {
    type Output = Array<T>;
    fn div(self, other: &Array<T>) -> Array<T> {
        Array::div(self, other)
    }
}

impl<T: Numeric> DivAssign<&Array<T>> for Array<T> {
    fn div_assign(&mut self, other: &Array<T>) {
        self.div_assign(other);
    }
}

impl<T: Numeric> Add<T> for &Array<T> {
    type Output = Array<T>;
    fn add(self, other: T) -> Array<T> {
        Array::add_scalar(self, other)
    }
}

impl<T: Numeric> AddAssign<T> for Array<T> {
    fn add_assign(&mut self, other: T) {
        self.add_assign_scalar(other);
    }
}

impl<T: Numeric> Sub<T> for &Array<T> {
    type Output = Array<T>;
    fn sub(self, other: T) -> Array<T> {
        Array::sub_scalar(self, other)
    }
}

impl<T: Numeric> SubAssign<T> for Array<T> {
    fn sub_assign(&mut self, other: T) {
        self.sub_assign_scalar(other);
    }
}

impl<T: Numeric> Mul<T> for &Array<T> {
    type Output = Array<T>;
    fn mul(self, other: T) -> Array<T> {
        Array::mul_scalar(self, other)
    }
}

impl<T: Numeric> MulAssign<T> for Array<T> {
    fn mul_assign(&mut self, other: T) {
        self.mul_assign_scalar(other);
    }
}

impl<T: Numeric> Div<T> for &Array<T> {
    type Output = Array<T>;
    fn div(self, other: T) -> Array<T> {
        Array::div_scalar(self, other)
    }
}

impl<T: Numeric> DivAssign<T> for Array<T> {
    fn div_assign(&mut self, other: T) {
        self.div_assign_scalar(other);
    }
}

impl<T: Numeric> Clone for Array<T> {
    fn clone(&self) -> Array<T> {
        Array {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}

/// Represents a slice on a single array dimension.
#[derive(PartialEq, Debug)]
pub enum Slice {
    // `x..y` - from to range.
    Range(Range<usize>),
    // `x..` - from range.
    RangeFrom(RangeFrom<usize>),
    // `..x` - to range.
    RangeTo(RangeTo<usize>),
    // `..` - whole axis range.
    RangeFull(RangeFull),
    // `x` - single index range.
    Index(usize),
}

impl From<usize> for Slice {
    fn from(index: usize) -> Slice {
        Slice::Index(index)
    }
}

impl From<Range<usize>> for Slice {
    fn from(range: Range<usize>) -> Slice {
        Slice::Range(range)
    }
}

impl From<RangeFrom<usize>> for Slice {
    fn from(range: RangeFrom<usize>) -> Slice {
        Slice::RangeFrom(range)
    }
}

impl From<RangeTo<usize>> for Slice {
    fn from(range: RangeTo<usize>) -> Slice {
        Slice::RangeTo(range)
    }
}

impl From<RangeFull> for Slice {
    fn from(range: RangeFull) -> Slice {
        Slice::RangeFull(range)
    }
}

/// Provides a convinient method to define array slice vector.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate neurust;
/// use neurust::prelude::*;
/// # fn main() {
/// let arr = Array::from_vec(
///    vec![1., 2., 3., 4., 5., 6., 7., 8.],
///    vec![2, 4]
/// );
///
/// arr.s(s![0, 1..3]);
/// arr.s(s![0, 1]);
/// arr.s(s![1.., 3]);
/// # }
/// ```
#[macro_export]
macro_rules! s {
    ([ $($stack:expr),* ] $num:expr) => {
        s![[$($stack, )* $crate::Slice::from($num)]]
    };
    ([ $($stack:expr),* ] $num:expr, $($middle:tt)*) => {
        s![[$($stack, )* $crate::Slice::from($num)] $($middle)*]
    };
    ([ $($stack:expr),* ] $num:expr, $($middle:tt),*) => {
        s![[$($stack, )* $crate::Slice::from($num)] $($middle),*]
    };
    ([ $($stack:expr),* ]) => {
        {
            let mut temp_vec = Vec::<$crate::Slice>::new();
            $(
                temp_vec.push($stack);
            )*
            temp_vec
        }
    };
    ($($tokens:tt)*) => {
        s!([] $($tokens)*)
    };
}

#[cfg(test)]
mod tests {
    pub use super::*;

    #[test]
    fn test_s() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        assert_eq!(
            arr.s(vec![Slice::Index(0), Slice::Range(0..2)]),
            ArrayView::new(
                &arr.data,
                vec![Slice::Index(0), Slice::Range(0..2)],
                &arr.shape
            )
        );
    }
}
