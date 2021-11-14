use std::cell::RefCell;
use std::rc::Rc;

use crate::variable::Variable;

pub trait OptimizerImpl {
    fn update(&mut self);
    fn set_params(&mut self, params: Vec<Rc<RefCell<Variable>>>);
    fn zero_grad(&mut self);
}

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

impl OptimizerImpl for SGD {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd() {
        let mut optim = SGD::new(0.1);

        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        optim.set_params(vec![x, y]);

        optim.zero_grad();
        optim.update();
    }
}
