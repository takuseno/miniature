use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

use crate::functions as F;
use crate::variable::Variable;

pub struct Linear {
    weight: Rc<RefCell<Variable>>,
    bias: Rc<RefCell<Variable>>,
    in_size: u32,
    out_size: u32,
}

impl Linear {
    pub fn new(in_size: u32, out_size: u32) -> Self {
        let weight = Rc::new(RefCell::new(Variable::new(vec![in_size, out_size])));
        let bias = Rc::new(RefCell::new(Variable::new(vec![1, out_size])));

        // randomly initialize weight
        let mut rng = rand::thread_rng();
        let weight_data = &mut weight.borrow_mut().data;
        for i in 0..(in_size * out_size) as usize {
            weight_data[i] = rng.gen();
        }

        // initialize bias with zeros
        bias.borrow_mut().zeros();

        Self {
            weight: weight.clone(),
            bias: bias,
            in_size: in_size,
            out_size: out_size,
        }
    }

    pub fn call(&self, x: Rc<RefCell<Variable>>) -> Rc<RefCell<Variable>> {
        let batch_size = x.borrow().shape[0];
        let h = F::matmul::matmul(x, self.weight.clone());
        let broadcasted_bias =
            F::broadcast::broadcast(self.bias.clone(), vec![batch_size, self.out_size]);
        F::add::add(h, broadcasted_bias)
    }

    pub fn get_params(&self) -> Vec<Rc<RefCell<Variable>>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;

    #[test]
    fn linear_forward_backward() {
        let fc1 = Linear::new(100, 200);
        let fc2 = Linear::new(200, 10);

        let x = Rc::new(RefCell::new(Variable::new(vec![32, 100])));
        let h = fc1.call(x);
        let output = fc2.call(h);

        assert_eq!(output.borrow().shape[0], 32);
        assert_eq!(output.borrow().shape[1], 10);

        backward(output);
    }
}
