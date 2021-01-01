#[cfg(test)]
mod tests {
    use neurust::{Array, Slice};

    #[test]
    #[should_panic]
    fn test_from_vec_wrong_shape() {
        Array::from_vec(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 1, 2],
        );
    }

    #[test]
    #[should_panic]
    fn test_from_vec_zero_shape() {
        Array::from_vec(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 1, 2, 3, 0],
        );
    }

    #[test]
    #[should_panic]
    fn test_new_zero_shape() {
        let _arr = Array::new(1., vec![2, 1, 2, 3, 0]);
    }

    #[test]
    #[should_panic]
    fn test_s_wrong_index_length() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr.s(vec![Slice::Index(0)]);
    }

    #[test]
    #[should_panic]
    fn test_s_index_out_of_bounds() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr.s(vec![Slice::Index(0), Slice::Index(3)]);
    }

    #[test]
    fn test_i() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        assert_eq!(arr.i(vec![1, 2]), 6.);
    }

    #[test]
    fn test_index() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        assert_eq!(arr[vec![1, 2]], 6.);
    }

    #[test]
    #[should_panic]
    fn test_index_wrong_index_length() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr[vec![1]];
    }

    #[test]
    #[should_panic]
    fn test_index_index_out_of_bounds() {
        let arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr[vec![1, 3]];
    }

    #[test]
    fn test_index_mut() {
        let mut arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr[vec![1, 2]] = 9.;

        assert_eq!(arr[vec![1, 2]], 9.);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_wrong_index_length() {
        let mut arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr[vec![1]] = 2.;
    }

    #[test]
    #[should_panic]
    fn test_index_mut_index_out_of_bounds() {
        let mut arr = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

        arr[vec![1, 3]] = 2.;
    }

    // Test arithmetic operations
    macro_rules! test_operator {
        (
            $name:ident, $function:ident, $function_assign:ident,
            $operator:tt, $operator_assign:tt, $expected_result:expr
        ) => {
            mod $name {
                use super::*;

                #[test]
                #[should_panic]
                fn test_wrong_shape() {
                    let a = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
                    let b = Array::from_vec(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 4]);

                    a.$function(&b);
                }

                #[test]
                #[should_panic]
                fn test_assign_wrong_shape() {
                    let mut a = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
                    let b = Array::from_vec(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 4]);

                    a.$function_assign(&b);
                }

                #[test]
                #[should_panic]
                fn test_wrong_shape_operator() {
                    let a = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
                    let b = Array::from_vec(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 4]);

                    let _ = &a + &b;
                }

                #[test]
                #[should_panic]
                fn test_assign_wrong_shape_operator() {
                    let mut a = Array::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
                    let b = Array::from_vec(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 4]);

                    a $operator_assign &b
                }

                #[test]
                fn test_method() {
                    let a = Array::from_vec(
                         vec![1., 2., 3., 4.],
                         vec![2, 1, 2]
                    );
                    let b = Array::from_vec(
                        vec![2., 3., 4., 5.],
                        vec![2, 1, 2]
                    );

                    let result = a.$function(&b);

                    assert_eq!(result, $expected_result)
                }

                #[test]
                fn test_method_assign() {
                    let mut a = Array::from_vec(
                         vec![1., 2., 3., 4.],
                         vec![2, 1, 2]
                    );
                    let b = Array::from_vec(
                        vec![2., 3., 4., 5.],
                        vec![2, 1, 2]
                    );

                    a.$function_assign(&b);

                    assert_eq!(a, $expected_result)
                }

                #[test]
                fn test_operator() {
                    let a = Array::from_vec(
                         vec![1., 2., 3., 4.],
                         vec![2, 1, 2]
                    );
                    let b = Array::from_vec(
                        vec![2., 3., 4., 5.],
                        vec![2, 1, 2]
                    );

                    let result = &a $operator &b;

                    assert_eq!(result, $expected_result)
                }

                #[test]
                fn test_operator_assign() {
                    let mut a = Array::from_vec(
                         vec![1., 2., 3., 4.],
                         vec![2, 1, 2]
                    );
                    let b = Array::from_vec(
                        vec![2., 3., 4., 5.],
                        vec![2, 1, 2]
                    );

                    a $operator_assign &b;

                    assert_eq!(a, $expected_result)
                }
            }
        }
    }

    test_operator!(
        test_add, add, add_assign, +, +=,
        Array::from_vec(vec![3., 5., 7., 9.], vec![2, 1, 2])
    );
    test_operator!(
        test_sub, sub, sub_assign, -, -=,
        Array::from_vec(vec![-1., -1., -1., -1.], vec![2, 1, 2])
    );
    test_operator!(
        test_mul, mul, mul_assign, *, *=,
        Array::from_vec(vec![2., 6., 12., 20.], vec![2, 1, 2])
    );
    test_operator!(
        test_div, div, div_assign, /, /=,
        Array::from_vec(vec![1. / 2., 2. / 3., 3. / 4., 4. / 5.], vec![2, 1, 2])
    );

    #[test]
    fn test_matmul() {
        let a = Array::from_vec(
            vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 3., 2., 1.],
            vec![2, 2, 3],
        );
        let b = Array::from_vec(
            vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.],
            vec![2, 3, 2],
        );

        let result = a.matmul(&b);

        assert_eq!(
            result,
            Array::from_vec(vec![22., 28., 49., 64., 22., 28., 14., 20.], vec![2, 2, 2],)
        )
    }

    #[test]
    fn test_matmul_broadcast() {
        let a = Array::from_vec(
            vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 3., 2., 1.],
            vec![2, 2, 3],
        );
        let b = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![1, 3, 2]);

        let result = a.matmul(&b);

        assert_eq!(
            result,
            Array::from_vec(vec![22., 28., 49., 64., 22., 28., 14., 20.], vec![2, 2, 2],)
        )
    }

    #[test]
    fn test_matmul_broadcast_smaller_shape() {
        let a = Array::from_vec(
            vec![1., 2., 3., 4., 5., 6., 1., 2., 3., 3., 2., 1.],
            vec![2, 2, 3],
        );
        let b = Array::from_vec(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);

        let result = a.matmul(&b);

        assert_eq!(
            result,
            Array::from_vec(vec![22., 28., 49., 64., 22., 28., 14., 20.], vec![2, 2, 2],)
        )
    }

    #[should_panic]
    #[test]
    fn test_matmul_wrong_broadcast() {
        let a = Array::new(1., vec![2, 2, 3]);
        let b = Array::new(1., vec![3, 3, 2]);

        a.matmul(&b);
    }
}
