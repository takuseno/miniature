use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Mean {}

impl Mean {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let output = outputs[0].borrow();
        assert_eq!(output.size(), 1);
    }
}

impl FunctionImpl for Mean {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        let mut sum = 0.0;
        for i in 0..x.size() {
            sum += x.data[i];
        }
        output.data[0] = sum / x.size() as f32;
    }

    fn backward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let output = outputs[0].borrow();

        let total_size = x.size();
        for i in 0..total_size {
            x.grad[i] += output.grad[0] / total_size as f32;
        }
    }

    fn get_name(&self) -> &str {
        "Mean"
    }
}
