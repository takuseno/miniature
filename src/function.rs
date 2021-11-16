use std::cell::RefCell;
use std::rc::Rc;

use crate::variable::Variable;

pub trait FunctionImpl {
    fn forward_impl(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]);
    fn backward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    );
    fn get_name(&self) -> &str;
}

impl std::fmt::Debug for dyn FunctionImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Function<{}>", self.get_name())
    }
}

#[derive(Debug)]
pub struct CgFunction {
    inputs: Vec<Rc<RefCell<Variable>>>,
    outputs: Vec<Rc<RefCell<Variable>>>,
    function_impl: Box<dyn FunctionImpl>,
}

impl CgFunction {
    pub fn new(
        inputs: Vec<Rc<RefCell<Variable>>>,
        outputs: Vec<Rc<RefCell<Variable>>>,
        function_impl: Box<dyn FunctionImpl>,
    ) -> Self {
        Self {
            inputs,
            outputs,
            function_impl,
        }
    }

    pub fn forward(&mut self) {
        self.function_impl.forward_impl(&self.inputs, &self.outputs);
    }

    pub fn backward(&mut self) {
        self.function_impl
            .backward_impl(&self.inputs, &self.outputs);
    }

    pub fn get_inputs(&self) -> &Vec<Rc<RefCell<Variable>>> {
        &self.inputs
    }
}
