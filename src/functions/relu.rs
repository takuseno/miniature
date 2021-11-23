use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct ReLu {}

impl ReLu {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        assert_eq!(x.shape, output.shape);
    }
}

impl FunctionImpl for ReLu {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.size() {
            output.data[i] = if x.data[i] > 0.0 { x.data[i] } else { 0.0 };
        }
    }

    fn backward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let output = outputs[0].borrow();

        for i in 0..x.size() {
            x.grad[i] += if x.data[i] > 0.0 { output.grad[i] } else { 0.0 };
        }
    }

    fn get_name(&self) -> &str {
        "ReLu"
    }
}
