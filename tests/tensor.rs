use neurust::{get_placeholder, get_variable, Array};
use std::collections::HashMap;

#[test]
fn test_variable() {
    let a = get_variable(Array::new(3., vec![2, 3, 2]));
    let b = get_variable(Array::new(2., vec![2, 2, 4]));

    assert_eq!(a.eval(None), Array::new(3., vec![2, 3, 2]));
    assert_eq!(a.grad(&a, None), Some(Array::new(1., vec![2, 3, 2])));
    assert_eq!(a.grad(&b, None), None);
}

#[test]
fn test_placeholder() {
    let a = get_placeholder("test_ph".to_owned());
    let b = get_variable(Array::new(2., vec![2, 2, 4]));
    let value = Array::new(3., vec![2, 2, 4]);
    let mut feed_dict = HashMap::new();
    feed_dict.insert("test_ph".to_owned(), &value);

    assert_eq!(
        a.eval(Some(feed_dict.clone())),
        Array::new(3., vec![2, 2, 4])
    );
    assert_eq!(
        a.grad(&a, Some(feed_dict.clone())),
        Some(Array::new(1., vec![2, 2, 4]))
    );
    assert_eq!(a.grad(&b, Some(feed_dict.clone())), None);
}

#[test]
#[should_panic]
fn test_placeholder_not_in_feed_dict() {
    let a = get_placeholder::<f32>("test_ph".to_owned());
    let feed_dict = HashMap::new();

    a.eval(Some(feed_dict.clone()));
}

macro_rules! test_tensor_operators {
    ($name:ident, $operator:tt, $result_eval:expr, $result_grad1:expr, $result_grad2:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn test_operator_eval(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));

                assert_eq!((&a $operator &b).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_eval_consume_left(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));

                assert_eq!((a $operator &b).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_eval_consume_right(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));

                assert_eq!((&a $operator b).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_eval_consume_both(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));

                assert_eq!((a $operator b).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_scalar_left(){
                let a = get_variable::<f32>(Array::new(1., vec![2, 2, 3]));

                assert_eq!((2. $operator &a).eval(None), $result_eval);
                assert_eq!((2. $operator a).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_scalar_right(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));

                assert_eq!((&a $operator 2.).eval(None), $result_eval);
                assert_eq!((a $operator 2.).eval(None), $result_eval);
            }

            #[test]
            fn test_operator_gradient(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));
                let c = get_variable(Array::new(2., vec![2, 2, 3]));
                let add = &a $operator &b;

                assert_eq!(add.grad(&a, None), Some($result_grad1));
                assert_eq!(add.grad(&b, None), Some($result_grad2));
                assert_eq!(add.grad(&c, None), None);
            }

            #[test]
            fn test_operator_scalar_gradient(){
                let a = get_variable(Array::new(1., vec![2, 2, 3]));
                let b = get_variable(Array::new(2., vec![2, 2, 3]));
                let add = &a $operator 2.;

                assert_eq!(add.grad(&a, None), Some($result_grad1));
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
    let a = get_variable(Array::new(1., vec![2, 3, 2]));
    let b = get_variable(Array::new(2., vec![2, 2, 4]));
    let c = get_variable(Array::new(2., vec![2, 2, 4]));
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
        let a = get_variable(Array::new(1., vec![2, 3, 2]));
        let b = get_variable(Array::new(1., vec![2, 3, 2]));
        let neg = -&a;

        assert_eq!(neg.eval(None), Array::new(-1., vec![2, 3, 2]));
        assert_eq!(neg.grad(&a, None), Some(Array::new(-1., vec![2, 3, 2])));
        assert_eq!(neg.grad(&b, None), None);
    }

    #[test]
    fn test_neg_consume_operator() {
        let a = get_variable(Array::new(1., vec![2, 3, 2]));
        let neg = -a;

        assert_eq!(neg.eval(None), Array::new(-1., vec![2, 3, 2]));
    }
}
