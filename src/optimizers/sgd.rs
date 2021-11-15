use std::cell::RefCell;
use std::rc::Rc;

use crate::optimizer::OptimizerImpl;
use crate::variable::Variable;

pub struct SGD {
    pub lr: f32,
}

impl OptimizerImpl for SGD {
    fn update(&mut self, params: &Vec<Rc<RefCell<Variable>>>) {
        for i in 0..params.len() {
            let mut param = params[i].borrow_mut();
            for j in 0..param.size() as usize {
                param.data[j] -= self.lr * param.grad[j];
            }
        }
    }
}
