use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Mean {}

impl Mean {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let output = outputs[0].borrow();
        assert_eq!(output.size(), 1);
    }
}

impl FunctionImpl for Mean {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        let mut sum = 0.0;
        for i in 0..x.size() as usize {
            sum += x.data[i];
        }
        output.data[0] = sum / x.size() as f32;
    }

    fn backward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let mut x = inputs[0].borrow_mut();
        let output = outputs[0].borrow();

        let total_size = x.size();
        for i in 0..total_size as usize {
            x.grad[i] += output.grad[0] / total_size as f32;
        }
    }

    fn get_name(&self) -> &str {
        "Mean"
    }
}

pub fn mean(x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(vec![1])));
    let function = Box::new(Mean {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x],
        outputs: vec![output.clone()],
        function_impl: function,
    }));
    cg_function.borrow_mut().forward();
    output.borrow_mut().set_parent(cg_function);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;

    #[test]
    fn mean_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = mean(x);
        backward(output);
    }
}
