use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;
use crate::function::FunctionImpl;
use crate::variable::Variable;

#[derive(Debug)]
pub struct Broadcast {
    shape: Vec<u32>,
}

impl Broadcast {
    fn validate(
        &mut self,
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        assert_eq!(inputs.len(), 1);
        assert_eq!(outputs.len(), 1);

        let x = inputs[0].borrow();
        let output = outputs[0].borrow();

        assert_eq!(x.shape.len(), self.shape.len());
        assert_eq!(output.shape, self.shape);
        for i in 0..x.shape.len() as usize {
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
        inputs: &Vec<Rc<RefCell<Variable>>>,
        outputs: &Vec<Rc<RefCell<Variable>>>,
    ) {
        self.validate(inputs, outputs);

        let x = inputs[0].borrow();
        let mut output = outputs[0].borrow_mut();

        // supports only batch dimension
        for i in 0..output.shape[0] {
            let offset = (i * x.size()) as usize;
            for j in 0..x.size() as usize {
                output.data[j + offset] = x.data[j];
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

        // supports only batch dimension
        for i in 0..x.size() as usize {
            for j in 0..output.shape[0] as usize {
                let offset = j * x.size() as usize;
                x.grad[i] += output.grad[i + offset];
            }
        }
    }

    fn get_name(&self) -> &str {
        "Broadcast"
    }
}

pub fn broadcast(x: Rc<RefCell<Variable>>, shape: Vec<u32>) -> Rc<RefCell<Variable>> {
    let output = Rc::new(RefCell::new(Variable::new(shape.clone())));
    let function = Box::new(Broadcast { shape: shape });
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
    fn broadcast_variables() {
        let x = Rc::new(RefCell::new(Variable::new(vec![1, 2, 3])));
        let output = broadcast(x, vec![3, 2, 3]);
        backward(output);
    }
}
