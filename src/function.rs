use std::rc::Rc;
use std::cell::RefCell;

use crate::variable::Variable;

pub trait FunctionImpl {
    fn forward_impl(&mut self, inputs: &Vec<Rc<RefCell<Variable>>>, outputs: &Vec<Rc<RefCell<Variable>>>);
    fn backward_impl(&mut self, inputs: &Vec<Rc<RefCell<Variable>>>, outputs: &Vec<Rc<RefCell<Variable>>>);
    fn get_name(&self) -> &str;
}

impl std::fmt::Debug for dyn FunctionImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Function<{}>", self.get_name())
    }
}

#[derive(Debug)]
pub struct CgFunction {
    pub inputs: Vec<Rc<RefCell<Variable>>>,
    pub outputs: Vec<Rc<RefCell<Variable>>>,
    pub function_impl: Box<dyn FunctionImpl>,
}

impl CgFunction {
    pub fn forward(&mut self) {
        self.function_impl.forward_impl(&self.inputs, &self.outputs);
    }

    pub fn backward(&mut self) {
        self.function_impl.backward_impl(&self.inputs, &self.outputs);
    }
}
