use rand::Rng;
use std::cell::RefCell;
use std::rc::Rc;

use crate::function::CgFunction;

#[derive(Debug)]
pub struct Variable {
    pub parent: Option<Rc<RefCell<CgFunction>>>,
    pub shape: Vec<u32>,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub need_grad: bool,
}

impl Variable {
    pub fn new(shape: Vec<u32>) -> Self {
        let mut size = 1;
        for i in 0..shape.len() {
            size *= shape[i];
        }

        let data = vec![0.0; size as usize];
        let grad = vec![0.0; size as usize];

        Self {
            parent: None,
            shape: shape,
            data: data,
            grad: grad,
            need_grad: true,
        }
    }

    pub fn rand(shape: Vec<u32>) -> Self {
        let mut variable = Self::new(shape);

        // randomly initialize weight
        let mut rng = rand::thread_rng();
        for i in 0..variable.size() as usize {
            variable.data[i] = rng.gen();
            variable.data[i] -= 0.5;
        }

        variable
    }

    pub fn size(&self) -> u32 {
        let mut size = 1;
        for i in 0..self.shape.len() {
            size *= self.shape[i];
        }
        size
    }

    pub fn set_data(&mut self, data: &[f32]) {
        assert_eq!(self.size() as usize, data.len());
        self.data.copy_from_slice(data);
    }

    pub fn set_grad(&mut self, grad: &[f32]) {
        assert_eq!(self.size() as usize, grad.len());
        self.grad.copy_from_slice(grad);
    }

    pub fn set_need_grad(&mut self, need_grad: bool) {
        self.need_grad = need_grad;
    }

    pub fn zeros(&mut self) {
        self.data.fill(0.0);
    }

    pub fn zero_grads(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn ones(&mut self) {
        self.data.fill(1.0);
    }

    pub fn one_grads(&mut self) {
        self.grad.fill(1.0);
    }

    pub fn set_parent(&mut self, parent: Rc<RefCell<CgFunction>>) {
        self.parent = Some(parent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_variable() {
        let mut variable = Variable::new(vec![1, 2, 3]);
        assert_eq!(variable.size(), 6);

        variable.data[0] = 1.0;
        variable.zeros();
        assert_eq!(variable.data[0], 0.0);

        variable.grad[0] = 1.0;
        variable.zero_grads();
        assert_eq!(variable.grad[0], 0.0);
    }
}
