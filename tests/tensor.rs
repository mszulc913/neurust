use neurust::linalg::utils::are_arrays_near_equal;
use neurust::{assert_arrays_rel_eq, Array, Tensor};
use std::collections::HashMap;

#[test]
fn test_variable() {
    let a = Tensor::new_variable(Array::new(3., vec![2, 3, 2]));
    let b = Tensor::new_variable(Array::new(2., vec![2, 2, 4]));

    assert_eq!(a.eval(None), Array::new(3., vec![2, 3, 2]));
    assert_eq!(a.grad(&a, None), Some(Array::new(1., vec![2, 3, 2])));
    assert_eq!(a.grad(&b, None), None);
}

#[test]
fn test_placeholder() {
    let a = Tensor::new_placeholder("test_ph".to_owned());
    let b = Tensor::new_variable(Array::new(2., vec![2, 2, 4]));
    let value = Array::new(3., vec![2, 2, 4]);
    let mut feed_dict = HashMap::new();
    feed_dict.insert("test_ph".to_owned(), &value);

    assert_eq!(a.eval(Some(&feed_dict)), Array::new(3., vec![2, 2, 4]));
    assert_eq!(
        a.grad(&a, Some(&feed_dict)),
        Some(Array::new(1., vec![2, 2, 4]))
    );
    assert_eq!(a.grad(&b, Some(&feed_dict)), None);
}

#[test]
#[should_panic]
fn test_placeholder_not_in_feed_dict() {
    let a = Tensor::<f32>::new_placeholder("test_ph".to_owned());
    let feed_dict = HashMap::new();

    a.eval(Some(&feed_dict));
}

macro_rules! test_tensor_operators {
    ($name:ident, $operator:tt, $result_eval:expr, $result_grad1:expr, $result_grad2:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn test_operator_eval(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));

                let res = (&a $operator &b).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_eval_consume_left(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));

                let res = (a $operator &b).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_eval_consume_right(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));

                let res = (&a $operator b).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_eval_consume_both(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));

                let res = (a $operator b).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_scalar_left(){
                let a = Tensor::<f32>::new_variable(Array::new(1., vec![2, 2, 3]));

                let res = (2. $operator &a).eval(None);
                let res_consume = (2. $operator a).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
                assert_arrays_rel_eq!(res_consume, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_scalar_right(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));

                let res = (&a $operator 2.).eval(None);
                let res_consume = (a $operator 2.).eval(None);

                assert_arrays_rel_eq!(res, $result_eval, 1e-7);
                assert_arrays_rel_eq!(res_consume, $result_eval, 1e-7);
            }

            #[test]
            fn test_operator_gradient(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));
                let c = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));
                let add = &a $operator &b;

                assert_arrays_rel_eq!(add.grad(&a, None).unwrap(), $result_grad1, 1e-7);
                assert_arrays_rel_eq!(add.grad(&b, None).unwrap(), $result_grad2, 1e-7);
                assert_eq!(add.grad(&c, None), None);
            }

            #[test]
            fn test_operator_scalar_gradient(){
                let a = Tensor::new_variable(Array::new(1., vec![2, 2, 3]));
                let b = Tensor::new_variable(Array::new(2., vec![2, 2, 3]));
                let add = &a $operator 2.;

                assert_arrays_rel_eq!(add.grad(&a, None).unwrap(), $result_grad1, 1e-7);
                assert_eq!(add.grad(&b, None), None);
            }
        }
    }
}

test_tensor_operators!(
    test_add, +, Array::new(3., vec![2, 2, 3]),
    Array::new(1., vec![2, 2, 3]), Array::new(1., vec![2, 2, 3])
);

test_tensor_operators!(
    test_sub, -, Array::new(-1., vec![2, 2, 3]),
    Array::new(-1., vec![2, 2, 3]), Array::new(-1., vec![2, 2, 3])
);

test_tensor_operators!(
    test_mul, *, Array::new(2., vec![2, 2, 3]),
    Array::new(2., vec![2, 2, 3]), Array::new(1., vec![2, 2, 3])
);

test_tensor_operators!(
    test_div, /, Array::new(0.5, vec![2, 2, 3]),
    Array::new(0.5, vec![2, 2, 3]), Array::new(-1. / 4., vec![2, 2, 3])
);

#[test]
fn test_matmul() {
    let a = Tensor::new_variable(Array::new(1., vec![2, 3, 2]));
    let b = Tensor::new_variable(Array::new(2., vec![2, 2, 4]));
    let c = Tensor::new_variable(Array::new(2., vec![2, 2, 4]));
    let matmul = a.matmul(&b);

    assert_eq!(matmul.eval(None), Array::new(4., vec![2, 3, 4]));
    assert_eq!(matmul.grad(&a, None), Some(Array::new(8., vec![2, 3, 2])));
    assert_eq!(matmul.grad(&b, None), Some(Array::new(3., vec![2, 2, 4])));
    assert_eq!(matmul.grad(&c, None), None);
}

mod test_neg {
    use super::*;

    #[test]
    fn test_neg_operator() {
        let a = Tensor::new_variable(Array::new(1., vec![2, 3, 2]));
        let b = Tensor::new_variable(Array::new(1., vec![2, 3, 2]));
        let neg = -&a;

        assert_eq!(neg.eval(None), Array::new(-1., vec![2, 3, 2]));
        assert_eq!(neg.grad(&a, None), Some(Array::new(-1., vec![2, 3, 2])));
        assert_eq!(neg.grad(&b, None), None);
    }

    #[test]
    fn test_neg_consume_operator() {
        let a = Tensor::new_variable(Array::new(1., vec![2, 3, 2]));
        let neg = -a;

        assert_eq!(neg.eval(None), Array::new(-1., vec![2, 3, 2]));
    }
}

#[test]
#[rustfmt::skip]
fn test_complex_example() {
    let a = Tensor::new_variable(Array::from_vec(
        vec![2., 3., 4., 5., 10., 12., 2., 6., 12., 10., 23., 12.],
        vec![2, 3, 2])
    );
    let b = Tensor::new_variable(Array::new(3., vec![2, 2, 4]));
    let c = Tensor::new_placeholder("test".to_owned());
    let ph_value = Array::new(17., vec![2, 3, 4]);
    let mut feed_dict = HashMap::new();
    feed_dict.insert("test".to_owned(), &ph_value);
    b.assign(&Array::new(3., vec![2, 2, 4]));

    let result = (a.matmul(&b) + 3.) * &c / 5.;

    assert_arrays_rel_eq!(
        &result.eval(Some(&feed_dict)),
        &Array::from_vec(
            vec![
                61.2, 61.2, 61.2, 61.2,
                102., 102., 102., 102.,
                234.6, 234.6, 234.6, 234.6,

                91.8, 91.8, 91.8, 91.8,
                234.6, 234.6, 234.6, 234.6,
                367.2, 367.2, 367.2, 367.2
            ],
            vec![2, 3, 4]
        ),
        1e-7
    );
    assert_arrays_rel_eq!(
        &result.grad(&a, Some(&feed_dict)).unwrap(),
        &Array::from_vec(
            vec![
                40.800003, 40.800003,
                40.800003, 40.800003,
                40.800003, 40.800003,

                40.800003, 40.800003,
                40.800003, 40.800003,
                40.800003, 40.800003,
            ],
            vec![2, 3, 2]
        ),
        1e-7
    );
    assert_arrays_rel_eq!(
        &result.grad(&b, Some(&feed_dict)).unwrap(),
        &Array::from_vec(
            vec![
                54.4, 54.4, 54.4, 54.4,
                68., 68., 68., 68.,

                125.8, 125.8 , 125.8, 125.8,
                95.200005, 95.200005, 95.200005, 95.200005
            ],
            vec![2, 2, 4]
        ),
        1e-7
    );
    assert_arrays_rel_eq!(
        &result.grad(&c, Some(&feed_dict)).unwrap(),
        &Array::from_vec(
            vec![
                3.6000001, 3.6000001, 3.6000001, 3.6000001,
                6., 6., 6., 6.,
                13.8, 13.8, 13.8, 13.8,

                5.4, 5.4, 5.4, 5.4,
                13.8, 13.8, 13.8, 13.8,
                21.6, 21.6, 21.6, 21.6
            ],
            vec![2, 3, 4]
        ),
        1e-7
    );
}
