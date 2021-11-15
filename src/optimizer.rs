use std::cell::RefCell;
use std::rc::Rc;

use crate::variable::Variable;

pub trait Optimizer {
    fn update(&mut self);
    fn set_params(&mut self, params: Vec<Rc<RefCell<Variable>>>);
    fn zero_grad(&mut self);
}
