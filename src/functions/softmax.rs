use std::cell::RefCell;
use std::rc::Rc;

use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Softmax {}

impl Softmax {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        // supports only 2-dim tensors
        assert_eq!(x.shape.len(), 2);
        assert_eq!(x.shape, output.shape);
    }
}

impl FunctionImpl for Softmax {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        // supports only 2-dim tensors
        for i in 0..x.shape[0] as usize {
            let offset = i * x.shape[1] as usize;
            let mut sum = 0.0;
            let mut max = x.data[offset];
            for j in 1..x.shape[1] as usize {
                max = if max > x.data[j + offset] {
                    max
                } else {
                    x.data[j + offset]
                };
            }
            for j in 0..x.shape[1] as usize {
                sum += (x.data[j + offset] - max).exp();
            }
            for j in 0..x.shape[1] as usize {
                output.data[j + offset] = (x.data[j + offset] - max).exp() / sum;
            }
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

        // supports only 2-dim tensors
        for i in 0..x.shape[0] as usize {
            let offset = i * x.shape[1] as usize;
            let mut sum = 0.0;
            for j in 0..x.shape[1] as usize {
                sum += output.data[j + offset] * output.grad[j + offset];
            }
            for j in 0..x.shape[1] as usize {
                x.grad[j + offset] += output.data[j + offset] * (output.grad[j + offset] - sum);
            }
        }
    }

    fn get_name(&self) -> &str {
        "Softmax"
    }
}
