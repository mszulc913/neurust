use crate::linalg::utils::check_shapes_broadcast;
use crate::linalg::Numeric;
use crate::Array;
use std::cmp::Ordering;

// Helper structure that while iterating returns corresponding
// (in terms of array broadcasting) data slices from two arrays.
pub(crate) struct BroadcastIterator<'a, T: Numeric> {
    array1: &'a Array<T>,
    array2: &'a Array<T>,
    slice1_len: usize,
    slice2_len: usize,
    broadcast_shape1: Vec<usize>,
    broadcast_shape2: Vec<usize>,
    broadcast_result_shape: Vec<usize>,
    current_index: Vec<usize>,
    done: bool,
}

impl<'a, T: Numeric> BroadcastIterator<'a, T> {
    pub fn new(
        array1: &'a Array<T>,
        array2: &'a Array<T>,
        trailing_dims: usize,
    ) -> BroadcastIterator<'a, T> {
        let max_shape_len = array1.shape.len().max(array2.shape.len());

        if max_shape_len < trailing_dims {
            panic!("Invalid `trailing_dims` parameter!. For given shapes it should be from <0, {}) interval.", max_shape_len)
        }

        let (broadcast_shape1, broadcast_shape2) =
            get_padded_broadcast_shapes(&array1.shape, &array2.shape, trailing_dims);
        let slice1_len = get_slice_len(&array1.shape, trailing_dims);
        let slice2_len = get_slice_len(&array2.shape, trailing_dims);

        check_shapes_broadcast(
            &array1.shape[..array1.shape.len() - trailing_dims],
            &array2.shape[..array2.shape.len() - trailing_dims],
        );

        let mut broadcast_result_shape = Vec::new();
        for i in 0..max_shape_len - trailing_dims {
            broadcast_result_shape.push(broadcast_shape1[i].max(broadcast_shape2[i]));
        }
        let current_index = vec![0; broadcast_result_shape.len()];

        BroadcastIterator {
            array1,
            array2,
            slice1_len,
            slice2_len,
            broadcast_shape1,
            broadcast_shape2,
            broadcast_result_shape,
            current_index,
            done: false,
        }
    }

    fn increment_broadcast_shape_iterator(&mut self) {
        *self.current_index.last_mut().unwrap() += 1;
        for i in (0..self.broadcast_result_shape.len()).rev() {
            if i == 0 && self.current_index[0] == self.broadcast_result_shape[0] {
                self.done = true;
            }
            if self.current_index[i] == self.broadcast_result_shape[i] && i != 0 {
                self.current_index[i] = 0;
                self.current_index[i - 1] += 1;
            }
        }
    }

    fn compute_slice_index(&self, shape: &[usize], slice_len: usize) -> usize {
        let mut idx = 0;
        let mut prod = slice_len;
        for i in (0..shape.len()).rev() {
            if self.current_index[i] < shape[i] {
                idx += self.current_index[i] * prod;
            }
            prod *= shape[i];
        }
        idx
    }
}

fn get_padded_broadcast_shapes(
    shape1: &[usize],
    shape2: &[usize],
    trailing_dims: usize,
) -> (Vec<usize>, Vec<usize>) {
    let shape1_len = shape1.len();
    let shape2_len = shape2.len();
    let shape1_trunc = &shape1[..shape1_len - trailing_dims];
    let shape2_trunc = &shape2[..shape2_len - trailing_dims];

    match shape1_len.cmp(&shape2_len) {
        Ordering::Equal => (shape1_trunc.to_vec(), shape2_trunc.to_vec()),
        Ordering::Greater => {
            let mut broadcast_shape2 = vec![1; shape1_len - shape2_len];
            broadcast_shape2.extend_from_slice(shape2_trunc);
            (shape1_trunc.to_vec(), broadcast_shape2)
        }
        Ordering::Less => {
            let mut broadcast_shape1 = vec![1; shape2_len - shape1_len];
            broadcast_shape1.extend_from_slice(shape1_trunc);
            (broadcast_shape1, shape2_trunc.to_vec())
        }
    }
}

fn get_slice_len(shape: &[usize], trailing_dims: usize) -> usize {
    if trailing_dims == 0 {
        1
    } else {
        shape[shape.len() - trailing_dims..].iter().product()
    }
}

impl<'a, T: Numeric> Iterator for BroadcastIterator<'a, T> {
    type Item = (&'a [T], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else if self.current_index.is_empty() {
            self.done = true;
            Some((self.array1.data.as_slice(), self.array2.data.as_slice()))
        } else {
            let slice1_index = self.compute_slice_index(&self.broadcast_shape1, self.slice1_len);
            let slice2_index = self.compute_slice_index(&self.broadcast_shape2, self.slice2_len);
            self.increment_broadcast_shape_iterator();
            Some((
                &self.array1.data[slice1_index..slice1_index + self.slice1_len],
                &self.array2.data[slice2_index..slice2_index + self.slice2_len],
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::broadcast::BroadcastIterator;
    use crate::Array;

    #[test]
    fn test_broadcast_iterator_new() {
        let a = Array::new(1., vec![3, 1, 2, 4, 5]);
        let b = Array::new(1., vec![3, 5, 1, 4, 5]);

        let iter = BroadcastIterator::new(&a, &b, 2);

        assert_eq!(iter.broadcast_shape1, vec![3, 1, 2]);
        assert_eq!(iter.broadcast_shape2, vec![3, 5, 1]);
        assert_eq!(iter.broadcast_result_shape, vec![3, 5, 2]);
        assert_eq!(iter.current_index, vec![0, 0, 0]);
    }

    #[test]
    fn test_broadcast_iterator_new_smaller_shape() {
        let a = Array::new(1., vec![3, 1, 2, 4, 5]);
        let b = Array::new(1., vec![5, 1, 4, 5]);

        let iter = BroadcastIterator::new(&a, &b, 2);

        assert_eq!(iter.broadcast_shape1, vec![3, 1, 2]);
        assert_eq!(iter.broadcast_shape2, vec![1, 5, 1]);
        assert_eq!(iter.broadcast_result_shape, vec![3, 5, 2]);
        assert_eq!(iter.current_index, vec![0, 0, 0]);
    }

    #[should_panic]
    #[test]
    fn test_broadcast_iterator_trailing_dims_too_big() {
        let a = Array::new(1., vec![3, 1, 2, 4, 5]);
        let b = Array::new(1., vec![5, 1, 4, 5]);

        BroadcastIterator::new(&a, &b, 5);
    }

    #[test]
    fn test_broadcast_iterator_next() {
        let a = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 1, 3]);
        let b = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![1, 3, 3]);

        let mut iter = BroadcastIterator::new(&a, &b, 1);

        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[1., 2., 3.][..])));
        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[4., 5., 6.][..])));
        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[7., 8., 9.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[1., 2., 3.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[4., 5., 6.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[7., 8., 9.][..])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_broadcast_iterator_next_longer_shape() {
        let a = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![1, 2, 1, 3]);
        let b = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8., 9.], vec![1, 3, 3]);

        let mut iter = BroadcastIterator::new(&a, &b, 1);

        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[1., 2., 3.][..])));
        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[4., 5., 6.][..])));
        assert_eq!(iter.next(), Some((&[1., 2., 3.][..], &[7., 8., 9.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[1., 2., 3.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[4., 5., 6.][..])));
        assert_eq!(iter.next(), Some((&[4., 5., 6.][..], &[7., 8., 9.][..])));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_broadcast_iterator_next_zero_trunc_dim() {
        let a = Array::from_vec(vec![1., 2., 3.], vec![1, 3]);
        let b = Array::from_vec(vec![1., 2., 3.], vec![3, 1]);

        let mut iter = BroadcastIterator::new(&a, &b, 0);

        assert_eq!(iter.next(), Some((&[1.][..], &[1.][..])));
        assert_eq!(iter.next(), Some((&[2.][..], &[1.][..])));
        assert_eq!(iter.next(), Some((&[3.][..], &[1.][..])));
        assert_eq!(iter.next(), Some((&[1.][..], &[2.][..])));
        assert_eq!(iter.next(), Some((&[2.][..], &[2.][..])));
        assert_eq!(iter.next(), Some((&[3.][..], &[2.][..])));
        assert_eq!(iter.next(), Some((&[1.][..], &[3.][..])));
        assert_eq!(iter.next(), Some((&[2.][..], &[3.][..])));
        assert_eq!(iter.next(), Some((&[3.][..], &[3.][..])));
        assert_eq!(iter.next(), None);
    }
}
