use std::cell::RefCell;
use std::rc::Rc;

use crate::optimizer::OptimizerImpl;
use crate::variable::Variable;

pub struct Sgd {
    pub lr: f32,
}

impl OptimizerImpl for Sgd {
    fn update(&mut self, params: &[Rc<RefCell<Variable>>]) {
        for param in params {
            let mut param = param.borrow_mut();
            for j in 0..param.size() as usize {
                param.data[j] -= self.lr * param.grad[j];
            }
        }
    }
}
