use std::cell::RefCell;
use std::rc::Rc;

use crate::optimizer::Optimizer;
use crate::variable::Variable;

pub struct SGD {
    params: Vec<Rc<RefCell<Variable>>>,
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            params: vec![],
            lr: lr,
        }
    }
}

impl Optimizer for SGD {
    fn set_params(&mut self, params: Vec<Rc<RefCell<Variable>>>) {
        for param in params.iter() {
            self.params.push(param.clone());
        }
    }

    fn update(&mut self) {
        for i in 0..self.params.len() {
            let mut param = self.params[i].borrow_mut();
            for j in 0..param.size() as usize {
                param.data[j] -= self.lr * param.grad[j];
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in self.params.iter() {
            param.borrow_mut().zero_grads();
        }
    }
}
