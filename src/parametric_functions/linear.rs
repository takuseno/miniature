use std::cell::RefCell;
use std::rc::Rc;

use crate::functions as F;
use crate::variable::Variable;

pub struct Linear {
    weight: Rc<RefCell<Variable>>,
    bias: Rc<RefCell<Variable>>,
    out_size: usize,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize) -> Self {
        let weight = Rc::new(RefCell::new(Variable::rand(vec![in_size, out_size])));
        let bias = Rc::new(RefCell::new(Variable::new(vec![1, out_size])));

        // initialize bias with zeros
        bias.borrow_mut().zeros();

        Self {
            weight,
            bias,
            out_size,
        }
    }

    pub fn call(&self, x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let batch_size = x.borrow().shape[0];
        let h = F::matmul(x, self.weight.clone());
        let broadcasted_bias = F::broadcast(self.bias.clone(), vec![batch_size, self.out_size]);
        F::add(h, broadcasted_bias)
    }

    pub fn get_params(&self) -> Vec<Rc<RefCell<Variable>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
