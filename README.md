# neurust

![build](https://github.com/mszulc913/neurust/workflows/build/badge.svg)

Deep learning library inspired by Python machine learning ecosystem written in *Rust*.

Purpose of this project is learn Rust by implementing deep learning framework that allow to
create *YOLO* algorithm without any other dependencies.
Since the project is educational, it tries not to use external dependencies
(only *num* crate for a numeric type trait and *CBLAS* bindings are used at the moment).

## Status
- Early development stage. 
- Not yet published on [crates.io](crates.io)
- No CUDA/OpenCL support.
- Lacks usage documentation.

## Dependencies
Matrix product is computed using [OpenBlas](https://github.com/xianyi/OpenBLAS), therefore 
make sure you have it installed, or the crate won't build.

## Highlights

### Arrays
N-dimensional array with support for slicing and basic operations like addition,
multiplication (using *CBLAS*) and element-wise multiplication:

```rust
use neurust::prelude::*;

let a = Array::from_vec(
    vec![
         1., 2., 3.,
         4., 5., 6.,
        
         1., 2., 3.,
         3., 2., 1.,
    ],
    vec![2, 2, 3]);
let b = Array::from_vec(
    vec![
        1., 2.,
        3., 4.,
        5., 6.,
        
        1., 2.,
        3., 4.,
        5., 6.
    ],
    vec![2, 3, 2]);

println!("{}", a.matmul(&b));
// outputs:
// [[[22, 28]
//   [49, 64]]
//
//  [[22, 28]
//   [14, 20]]] shape=[2, 2, 2]


```

### Tensors
A computational graph's node that can be evaluated and differentiated using reverse accumulation:

```rust
use neurust::prelude::*;

let a = Tensor::new_variable(Array::from_vec(
    vec![
        0., 1., 2.,
        3., 4., 5.
    ],
    vec![2, 3]));
let b = Tensor::new_variable(Array::from_vec(
    vec![
        4., 5., 6.
    ],
    vec![3, 1]));
let mul = a.matmul(&b); // matrix product of a and b

println!("{}", mul.grad(&b, None).unwrap());
// outputs:
// [[3]
//  [5]
//  [7]] shape=[3, 1]
```

### To be implemented
- Neural networks API (optimizers, layers, initializers)
- More tensor operators
- I/O helpers