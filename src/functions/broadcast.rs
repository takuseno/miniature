use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Broadcast {
    pub shape: Vec<usize>,
}

impl Broadcast {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        assert_eq!(x.shape.len(), self.shape.len());
        assert_eq!(output.shape, self.shape);
        for i in 0..x.shape.len() {
            if x.shape[i] != 1 {
                assert_eq!(x.shape[i], self.shape[i]);
            } else {
                assert_eq!(x.shape[i], 1);
            }
        }
    }
}

impl FunctionImpl for Broadcast {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        // supports only batch dimension
        for i in 0..output.shape[0] {
            let offset = i * x.size();
            for j in 0..x.size() {
                output.data[j + offset] = x.data[j];
            }
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

        // supports only batch dimension
        for i in 0..x.size() {
            for j in 0..output.shape[0] {
                let offset = j * x.size();
                x.grad[i] += output.grad[i + offset];
            }
        }
    }

    fn get_name(&self) -> &str {
        "Broadcast"
    }
}
