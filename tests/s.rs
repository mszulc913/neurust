use neurust::{s, Slice};

#[test]
fn test_slice_macro() {
    assert_eq!(s![], Vec::<Slice>::new());
    assert_eq!(s![1], vec![Slice::Index(1)]);
    assert_eq!(s![1, 2], vec![Slice::Index(1), Slice::Index(2)]);
    assert_eq!(s![1..2], vec![Slice::Range(1..2)]);
    assert_eq!(s![..2], vec![Slice::RangeTo(..2)]);
    assert_eq!(s![1..], vec![Slice::RangeFrom(1..)]);
    assert_eq!(s![..], vec![Slice::RangeFull(..)]);
    assert_eq!(
        s![1, 1..2, 2.., 3, .., ..],
        vec![
            Slice::Index(1),
            Slice::Range(1..2),
            Slice::RangeFrom(2..),
            Slice::Index(3),
            Slice::RangeFull(..),
            Slice::RangeFull(..)
        ]
    );
}
