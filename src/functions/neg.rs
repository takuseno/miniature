use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Neg {}

impl Neg {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = inputs[0].borrow();

        assert_eq!(x.shape, output.shape);
    }
}

impl FunctionImpl for Neg {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.size() as usize {
            output.data[i] = -x.data[i];
        }
    }

    fn backward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let output = outputs[0].borrow();

        for i in 0..x.size() as usize {
            x.grad[i] -= output.grad[i];
        }
    }

    fn get_name(&self) -> &str {
        "Neg"
    }
}