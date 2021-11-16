use std::cell::RefCell;
use std::rc::Rc;

use crate::variable::Variable;

pub trait OptimizerImpl {
    fn update(&mut self, params: &[Rc<RefCell<Variable>>]);
}

pub struct Optimizer {
    params: Vec<Rc<RefCell<Variable>>>,
    optimizer_impl: Box<dyn OptimizerImpl>,
}

impl Optimizer {
    pub fn new(optimizer_impl: Box<dyn OptimizerImpl>) -> Self {
        Self {
            params: vec![],
            optimizer_impl,
        }
    }

    pub fn set_params(&mut self, params: Vec<Rc<RefCell<Variable>>>) {
        for param in params.iter() {
            self.params.push(param.clone());
        }
    }

    pub fn update(&mut self) {
        self.optimizer_impl.update(&self.params);
    }

    pub fn zero_grad(&mut self) {
        for param in self.params.iter() {
            param.borrow_mut().zero_grads();
        }
    }
}
