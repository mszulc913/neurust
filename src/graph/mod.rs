pub(crate) mod arithmetic;
pub(crate) mod math;

use crate::linalg::{Array, Numeric};
use std::any::{type_name, Any};
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

// Computational graph's node.
// TODO: Store shapes in structs.
pub trait GraphOp<T: Numeric> {
    // Computes value of the operator recursively traversing through the graph
    // with usage of a cache map.
    fn compute(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T>;

    // Computes gradient of some variable `X` w.r.t. `dependant_node` given gradient
    // of `X` w.r.t. `self` using chain rule.
    fn compute_accum_grad(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>>;

    // Returns name of the operation.
    fn get_name(&self) -> &str {
        "UnnamedOp"
    }

    // Returns references to input nodes.
    fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
        None
    }

    // Returns computed value of the node.
    // This either fetches the value from `compute_cache` or computes it via `compute()`.
    fn value(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        let key = self.ref_as_usize();
        if let Some(cached_value) = compute_cache.get(&key) {
            cached_value.clone()
        } else {
            let value = self.compute(feed_dict, compute_cache);
            let return_value = value.clone();
            compute_cache.insert(key, value);
            return_value
        }
    }

    // Computes node's address as `usize`.
    fn ref_as_usize(&self) -> usize {
        self as *const _ as *const usize as usize
    }

    // Evaluates the operation.
    fn eval(&self, feed_dict: Option<HashMap<String, &Array<T>>>) -> Array<T> {
        let mut compute_cache = HashMap::<usize, Array<T>>::new();
        let feed_unwrapped = feed_dict.unwrap_or_default();
        self.value(&feed_unwrapped, &mut compute_cache)
    }

    // Computes gradient of the node (`self`) w.r.t. operation (variable) `node`.
    fn grad(
        &self,
        node: &dyn GraphOp<T>,
        feed_dict: Option<HashMap<String, &Array<T>>>,
    ) -> Option<Array<T>> {
        let mut compute_cache = HashMap::<usize, Array<T>>::new();
        let mut accum_grad_map = HashMap::<usize, Array<T>>::new();
        let mut stack = Vec::<(Rc<dyn GraphOp<T>>, Rc<dyn GraphOp<T>>)>::new();
        let feed_unwrapped = feed_dict.unwrap_or_default();

        let accum_grad_self = Array::<T>::new(
            T::one(),
            self.compute(&feed_unwrapped, &mut compute_cache)
                .get_shape(),
        );
        if self.ref_as_usize() == node.ref_as_usize() {
            return Some(accum_grad_self);
        }

        let phantom_parent: Rc<dyn GraphOp<T>> = Rc::new(WrapperOp::<T>::new(self.as_trait()));
        accum_grad_map.insert(phantom_parent.ref_as_usize(), accum_grad_self);

        let children = self.get_inputs().unwrap_or_default();
        for child in children {
            stack.push((Rc::clone(&child), Rc::clone(&phantom_parent)))
        }

        while let Some((current_node, current_parrent)) = stack.pop() {
            let parrent_grad = current_parrent.compute_accum_grad(
                &feed_unwrapped,
                &mut compute_cache,
                current_node.as_ref(),
                &accum_grad_map[&current_parrent.ref_as_usize()],
            );
            if let Some(grad) = parrent_grad {
                if let Some(accum_grad) = accum_grad_map.get_mut(&current_node.ref_as_usize()) {
                    *accum_grad += &grad;
                } else {
                    accum_grad_map.insert(current_node.ref_as_usize(), grad);
                }
            }
            let children = current_node.get_inputs().unwrap_or_default();
            if current_node.ref_as_usize() != node.ref_as_usize() && !children.is_empty() {
                for child in children {
                    stack.push((Rc::clone(&child), Rc::clone(&current_node)))
                }
            }
        }
        accum_grad_map.remove(&node.ref_as_usize())
    }

    // Returns reference to a particular trait object as `GraphOp<T>`. This is needed
    // to provide 'gradient()' default implementation.
    fn as_trait(&self) -> &dyn GraphOp<T>;
}

impl<T: Numeric> PartialEq for dyn GraphOp<T> {
    fn eq(&self, other: &Self) -> bool {
        let inputs1 = self.get_inputs().unwrap_or_default();
        let inputs2 = other.get_inputs().unwrap_or_default();
        let mut is_equal = self.type_id() == other.type_id() && inputs1.len() == inputs2.len();
        if is_equal {
            for (input1, input2) in inputs1.iter().zip(inputs2) {
                if input1.ref_as_usize() != input2.ref_as_usize() {
                    is_equal = false;
                    break;
                }
            }
        }
        is_equal
    }
}

impl<T: Numeric> fmt::Debug for dyn GraphOp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            format!(
                "Op: <{}>, inputs: {:?}",
                type_name::<Self>(),
                self.get_inputs()
            )
            .as_str(),
        )
    }
}

// Provides a wrapper around other node given its reference.
pub struct WrapperOp<'a, T: Numeric> {
    input: &'a dyn GraphOp<T>,
}

impl<'a, T: Numeric> WrapperOp<'a, T> {
    pub fn new(input: &dyn GraphOp<T>) -> WrapperOp<T> {
        WrapperOp { input }
    }
}

impl<'a, T: Numeric> GraphOp<T> for WrapperOp<'a, T> {
    fn compute(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        self.input.value(feed_dict, compute_cache)
    }

    fn compute_accum_grad(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        compute_cache: &mut HashMap<usize, Array<T>>,
        dependant_node: &dyn GraphOp<T>,
        grad: &Array<T>,
    ) -> Option<Array<T>> {
        self.input
            .compute_accum_grad(feed_dict, compute_cache, dependant_node, grad)
    }

    fn get_name(&self) -> &str {
        "WrapperOp"
    }

    fn get_inputs(&self) -> Option<Vec<Rc<dyn GraphOp<T>>>> {
        None
    }

    fn as_trait(&self) -> &dyn GraphOp<T> {
        self as &dyn GraphOp<T>
    }
}

// Placeholder for values to be supplied later.
pub struct Placeholder {
    pub id: String,
}

impl Placeholder {
    pub fn new(id: String) -> Placeholder {
        Placeholder { id }
    }
}

impl<T: Numeric> GraphOp<T> for Placeholder {
    fn compute(
        &self,
        feed_dict: &HashMap<String, &Array<T>>,
        _: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        if let Some(value) = feed_dict.get(&self.id) {
            (*value).clone()
        } else {
            panic!("Value not found in feed_dict: {}", self.id)
        }
    }

    fn compute_accum_grad(
        &self,
        _: &HashMap<String, &Array<T>>,
        _: &mut HashMap<usize, Array<T>>,
        _: &dyn GraphOp<T>,
        _: &Array<T>,
    ) -> Option<Array<T>> {
        None
    }

    fn get_name(&self) -> &str {
        "PlaceholderOp"
    }

    fn as_trait(&self) -> &dyn GraphOp<T> {
        self as &dyn GraphOp<T>
    }
}

// Shared and persistent data stored in a operational memory.
pub struct Variable<T: Numeric> {
    data: Array<T>,
}

impl<T: Numeric> Variable<T> {
    pub fn new(init_value: Array<T>) -> Variable<T> {
        Variable { data: init_value }
    }
}

impl<T: Numeric> GraphOp<T> for Variable<T> {
    fn compute(
        &self,
        _: &HashMap<String, &Array<T>>,
        _: &mut HashMap<usize, Array<T>>,
    ) -> Array<T> {
        self.data.clone()
    }

    fn compute_accum_grad(
        &self,
        _: &HashMap<String, &Array<T>>,
        _: &mut HashMap<usize, Array<T>>,
        _: &dyn GraphOp<T>,
        _: &Array<T>,
    ) -> Option<Array<T>> {
        None
    }

    fn get_name(&self) -> &str {
        "VariableOp"
    }

    fn as_trait(&self) -> &dyn GraphOp<T> {
        self as &dyn GraphOp<T>
    }
}
