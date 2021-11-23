use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Argmax {}

impl Argmax {
    fn validate(&mut self, inputs: &[Rc<RefCell<Variable>>], outputs: &[Rc<RefCell<Variable>>]) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        // supports only 2-dim tensors
        assert_eq!(x.shape.len(), 2);
        assert_eq!(output.shape.len(), 1);
        assert_eq!(x.shape[0], output.shape[0]);
    }
}

impl FunctionImpl for Argmax {
    fn forward_impl(
        &mut self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.shape[0] {
            let offset = i * x.shape[1];
            let mut max = x.data[offset];
            let mut max_index = 0.0;
            for j in 1..x.shape[1] {
                if x.data[j + offset] > max {
                    max = x.data[j + offset];
                    max_index = j as f32;
                }
            }
            output.data[i] = max_index;
        }
    }

    fn backward_impl(
        &mut self,
        _inputs: &[Rc<RefCell<Variable>>],
        _outputs: &[Rc<RefCell<Variable>>],
    ) {
        panic!("Argmax does not support backward.")
    }

    fn get_name(&self) -> &str {
        "Argmax"
    }
}
