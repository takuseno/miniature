use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Add {}

impl Add {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let y = inputs[0].borrow();
        let output = outputs[0].borrow();

        assert_eq!(x.shape, y.shape);
        assert_eq!(x.shape, output.shape);
    }
}

impl FunctionImpl for Add {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let y = inputs[1].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.size() {
            output.data[i] = x.data[i] + y.data[i];
        }
    }

    fn backward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let mut y = inputs[1].borrow_mut();
        let output = outputs[0].borrow();

        for i in 0..x.size() {
            x.grad[i] += output.grad[i];
            y.grad[i] += output.grad[i];
        }
    }

    fn get_name(&self) -> &str {
        "Add"
    }
}
