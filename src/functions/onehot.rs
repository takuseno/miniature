use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Onehot {
    pub num_classes: u32,
}

impl Onehot {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        // supports only 1-dim vectors
        assert_eq!(x.shape.len(), 1);
        assert_eq!(output.shape.len(), 2);
        assert_eq!(output.shape[0], x.shape[0]);
        assert_eq!(output.shape[1], self.num_classes);
    }
}

impl FunctionImpl for Onehot {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.size() as usize {
            let offset = i * self.num_classes as usize;
            for j in 0..self.num_classes as usize {
                output.data[j + offset] = if x.data[i] == j as f32 { 1.0 } else { 0.0 }
            }
        }
    }

    fn backward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        panic!("Onehot does not support backward.");
    }

    fn get_name(&self) -> &str {
        "Onehot"
    }
}
