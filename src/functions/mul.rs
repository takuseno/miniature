use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Mul {}

impl Mul {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);
        let x = inputs[0].borrow();
        let y = inputs[0].borrow();
        let output = inputs[0].borrow();
        assert_eq!(x.shape, y.shape);
        assert_eq!(x.shape, output.shape);
    }
}

impl FunctionImpl for Mul {
    fn forward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let y = inputs[1].borrow();
        let mut output = outputs[0].borrow_mut();

        for i in 0..x.size() as usize {
            output.data[i] = x.data[i] * y.data[i];
        }
    }

    fn backward_impl(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(&inputs, &outputs);

        let mut x = inputs[0].borrow_mut();
        let mut y = inputs[1].borrow_mut();
        let output = outputs[0].borrow();

        for i in 0..x.size() as usize {
            x.grad[i] = x.data[i] * output.grad[i];
            y.grad[i] = y.data[i] * output.grad[i];
        }
    }

    fn get_name(&self) -> &str {
        "Mul"
    }
}

pub fn mul(x: Rc<RefCell<Variable>>, y: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(x.borrow().shape.clone())));
    let function = Box::new(Mul {});
    let cg_function = Rc::new(RefCell::new(CgFunction {
        inputs: vec![x, y],
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
    fn mul_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let y = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let h = mul(x, y);
        let z = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = mul(h, z);
        backward(output);
    }
}
